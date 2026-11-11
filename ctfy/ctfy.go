// Package ctfy implements the small, self-contained part of the CTFY-0001
// agent protocol the prtseek runner needs: reading the briefing the platform
// drops in $CTFY_WORK_DIR, talking to the platform API with the bearer it hands
// us, and writing result.json back.
//
// It deliberately depends on nothing in the rest of prtseek, so the protocol
// contract can be read in one file and tested without a model or a network.
package ctfy

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// SchemaVersion is the wire version this client speaks (CTFY-0001 §5).
const SchemaVersion = 3

// DefaultWorkDir is where the platform puts the briefing when CTFY_WORK_DIR is
// unset. The platform also starts the container with this as its cwd.
const DefaultWorkDir = "/ctfy/work"

// Run modes (CTFY-0001 §2).
const (
	ModeInstance  = "instance"
	ModeQA        = "qa"
	ModeDefence   = "defence"
	ModeAWD       = "awd"
	ModeEngage    = "engagement"
)

// Question is one milestone as it appears in task.json.
type Question struct {
	ID      string `json:"id"`
	Prompt  string `json:"prompt"`
	Mode    string `json:"mode,omitempty"`
	Choices []string `json:"choices,omitempty"`
	// Requires lists the question ids that must be solved before this prompt is
	// even visible. Gated questions arrive with an empty Prompt until unlocked.
	Requires []string `json:"requires,omitempty"`
}

// Attachment is one file the platform copied into attachments/.
type Attachment struct {
	Name string `json:"name"`
	Size int64  `json:"size"`
	// Path is null when the copy was skipped (the 4 MiB cap); README then gives
	// the API call that downloads it.
	Path *string `json:"path"`
}

// TargetEndpoint is one address the player is shown for the target.
type TargetEndpoint struct {
	Label string `json:"label"`
	Type  string `json:"type"` // http, https, tcp, udp, ssh
	URL   string `json:"url"`
}

// Challenge is the challenge block of task.json.
type Challenge struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description"`
}

// ResultSpec tells us where to write the result, and whether stdout is parsed.
type ResultSpec struct {
	Path string `json:"path"`
	// StdoutFallback is true only for the platform's own images. For a
	// user-supplied image it is false and stdout is never read (CTFY-0001 §3).
	StdoutFallback bool `json:"stdout_fallback"`
}

// Limits are the resource bounds this run actually has.
type Limits struct {
	MemoryBytes int64 `json:"memory_bytes,omitempty"`
	CPUs        int   `json:"cpus,omitempty"`
	Pids        int   `json:"pids,omitempty"`
}

// Task is $CTFY_WORK_DIR/task.json — the machine-readable contract.
type Task struct {
	SchemaVersion int    `json:"schema_version"`
	RunMode       string `json:"run_mode"`
	Challenge     Challenge `json:"challenge"`
	CompetitionID string `json:"competition_id"`
	Questions     []Question `json:"questions"`
	// GatedQuestions counts the questions whose prompt the platform withheld
	// until an earlier one is answered. >0 means re-read the challenge after
	// every correct answer (CTFY-0001 §2).
	GatedQuestions int `json:"gated_questions"`
	Scope          []string `json:"scope"`
	RulesOfEngagement []string `json:"rules_of_engagement"`
	Deliverable    string `json:"deliverable"`
	PlatformURL    string `json:"platform_url"`
	// MaxTurns is null when no budget was set — never 0.
	MaxTurns       *int       `json:"max_turns"`
	Result         ResultSpec `json:"result"`
	TargetInstance string     `json:"target_instance"`
	TargetEndpoints []TargetEndpoint `json:"target_endpoints"`
	Attachments    []Attachment `json:"attachments"`
	// DryRun is true for a conformance check: no credentials, no platform.
	DryRun         bool       `json:"dry_run"`
	Limits         *Limits    `json:"limits"`
}

// Env is everything the platform passes in the environment (CTFY-0001 §1.1).
type Env struct {
	PlatformURL   string
	Token         string
	ChallengeID   string
	CompetitionID string
	TargetInstance string
	TargetURL     string
	MaxTurns      *int
	WorkDir       string

	LLMFlavor  string
	LLMAPIKey  string
	LLMModel   string
	LLMBaseURL string

	DryRun bool
}

// LoadEnv reads and normalises the CTFY environment. Nothing here validates
// that a key is present — that is the caller's job, and an empty key is a real
// state (CTFY-0001 §1.1), not an error to paper over.
func LoadEnv() Env {
	e := Env{
		PlatformURL:    strings.TrimRight(strings.TrimSpace(os.Getenv("CTFY_PLATFORM_URL")), "/"),
		Token:          strings.TrimSpace(os.Getenv("CTFY_TOKEN")),
		ChallengeID:    strings.TrimSpace(os.Getenv("CTFY_EVAL_CHALLENGE_ID")),
		CompetitionID:  strings.TrimSpace(os.Getenv("CTFY_EVAL_COMPETITION_ID")),
		TargetInstance: strings.TrimSpace(os.Getenv("CTFY_EVAL_TARGET_INSTANCE")),
		TargetURL:      strings.TrimSpace(os.Getenv("CTFY_TARGET_URL")),
		WorkDir:        strings.TrimSpace(os.Getenv("CTFY_WORK_DIR")),
		LLMFlavor:      strings.ToLower(strings.TrimSpace(os.Getenv("LLM_FLAVOR"))),
		LLMAPIKey:      os.Getenv("LLM_API_KEY"),
		LLMModel:       strings.TrimSpace(os.Getenv("LLM_MODEL")),
		LLMBaseURL:     strings.TrimRight(strings.TrimSpace(os.Getenv("LLM_BASE_URL")), "/"),
		DryRun:         os.Getenv("CTFY_EVAL_DRY_RUN") == "1",
	}
	if e.WorkDir == "" {
		e.WorkDir = DefaultWorkDir
	}
	if v := strings.TrimSpace(os.Getenv("CTFY_EVAL_MAX_TURNS")); v != "" {
		var n int
		if _, err := fmt.Sscanf(v, "%d", &n); err == nil && n > 0 {
			e.MaxTurns = &n
		}
	}
	// The platform publishes the model id under several Claude-Code-compatible
	// slots too; fall back to those when LLM_MODEL itself is absent.
	if e.LLMModel == "" {
		for _, k := range []string{"ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "OPENAI_MODEL"} {
			if v := strings.TrimSpace(os.Getenv(k)); v != "" {
				e.LLMModel = v
				break
			}
		}
	}
	// A base URL may arrive under the SDK's own name when LLM_BASE_URL is unset.
	if e.LLMBaseURL == "" {
		for _, k := range []string{"ANTHROPIC_BASE_URL", "OPENAI_BASE_URL"} {
			if v := strings.TrimRight(strings.TrimSpace(os.Getenv(k)), "/"); v != "" {
				e.LLMBaseURL = v
				break
			}
		}
	}
	return e
}

// ReadTask loads task.json from the working directory.
func ReadTask(workDir string) (*Task, error) {
	path := filepath.Join(workDir, "task.json")
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	var t Task
	if err := json.Unmarshal(b, &t); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	if t.SchemaVersion > SchemaVersion {
		return nil, fmt.Errorf("task.json schema_version %d is newer than this client (%d); refusing to misread it", t.SchemaVersion, SchemaVersion)
	}
	if t.MaxTurns == nil {
		t.MaxTurns = nil // explicit: null means "no budget", never 0
	}
	return &t, nil
}

// ReadBriefing returns README.md verbatim — the prose rendering of task.json,
// which §3.2 says a conforming agent may simply read and act on.
func ReadBriefing(workDir string) string {
	b, err := os.ReadFile(filepath.Join(workDir, "README.md"))
	if err != nil {
		return ""
	}
	return string(b)
}

// Result is what we write to result.json (CTFY-0001 §3). Every field is
// optional and the platform ignores what it does not need.
type Result struct {
	SchemaVersion int               `json:"schema_version"`
	Answers       map[string]string `json:"answers,omitempty"`
	// Solved is advisory only: the platform writes its own record of what was
	// actually submitted. Listing an id here that was never submitted is not
	// true and is reported as such.
	Solved         []string `json:"solved,omitempty"`
	FullySolved    bool     `json:"fully_solved"`
	Turns          int      `json:"turns,omitempty"`
	ToolCallsMade  int      `json:"tool_calls_made,omitempty"`
	InputTokens    int      `json:"input_tokens,omitempty"`
	OutputTokens   int      `json:"output_tokens,omitempty"`
	StopReason     string   `json:"stop_reason,omitempty"`
	Writeup        string   `json:"writeup,omitempty"`
}

// WriteResult writes result.json atomically into the working directory. A
// missing file is merely advice on a clean exit, but writing one is how a run
// hands over answers it never submitted itself (CTFY-0001 §3.2).
func WriteResult(workDir string, r Result) error {
	r.SchemaVersion = SchemaVersion
	if len(r.Answers) == 0 {
		r.Answers = nil
	}
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	path := filepath.Join(workDir, "result.json")
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// APIError is a non-2xx answer from the platform, kept structured so a caller
// can branch on `code` (e.g. competition_paused, token_challenge_forbidden).
type APIError struct {
	Status  int
	Code    string          `json:"code"`
	Message string          `json:"message"`
	Body    json.RawMessage `json:"-"`
}

func (e *APIError) Error() string {
	msg := strings.TrimSpace(e.Message)
	if msg == "" {
		msg = strings.TrimSpace(string(e.Body))
	}
	if len(msg) > 300 {
		msg = msg[:300] + "…"
	}
	if e.Code != "" {
		return fmt.Sprintf("platform %d %s: %s", e.Status, e.Code, msg)
	}
	return fmt.Sprintf("platform %d: %s", e.Status, msg)
}

// Client is a minimal platform API client (CTFY-0001 §1.2).
type Client struct {
	baseURL string
	token   string
	http    *http.Client
}

// NewClient builds a client. An empty base URL or token yields a client whose
// calls fail explicitly — that is the correct state on a dry run, where there
// is no platform to talk to.
func NewClient(platformURL, token string) *Client {
	return &Client{
		baseURL: strings.TrimRight(platformURL, "/"),
		token:   token,
		http:    &http.Client{Timeout: 45 * time.Second},
	}
}

// Available reports whether this client can actually reach a platform.
func (c *Client) Available() bool { return c != nil && c.baseURL != "" && c.token != "" }

func (c *Client) do(method, path string, body any, out any) error {
	if !c.Available() {
		return errors.New("no platform credentials in this run (dry run or unfunded)")
	}
	var rd io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return err
		}
		rd = bytes.NewReader(b)
	}
	req, err := http.NewRequest(method, c.baseURL+"/api/v1"+path, rd)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+c.token)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		apiErr := &APIError{Status: resp.StatusCode, Body: json.RawMessage(raw)}
		var shape struct {
			Code    string `json:"code"`
			Message string `json:"message"`
			Detail  string `json:"detail"`
		}
		if json.Unmarshal(raw, &shape) == nil {
			apiErr.Code, apiErr.Message = shape.Code, shape.Message
			if apiErr.Message == "" {
				apiErr.Message = shape.Detail
			}
		}
		return apiErr
	}
	if out != nil && len(raw) > 0 {
		if err := json.Unmarshal(raw, out); err != nil {
			return fmt.Errorf("decode %s %s: %w", method, path, err)
		}
	}
	return nil
}

// Me is GET /api/v1/me — the cheapest way to confirm the token works.
func (c *Client) Me(out any) error { return c.do(http.MethodGet, "/me", nil, out) }

// ChallengeDetail mirrors the public ChallengeInfo the platform returns.
type ChallengeDetail struct {
	ID        string     `json:"id"`
	Name      string     `json:"name"`
	Category  string     `json:"category"`
	Difficulty string    `json:"difficulty"`
	Questions []Question `json:"questions"`
	IsQAOnly  bool       `json:"is_qa_only"`
}

// GetChallenge re-reads a challenge. Required after each correct answer when
// gated questions exist: their prompts only materialise once unlocked.
func (c *Client) GetChallenge(id string) (*ChallengeDetail, error) {
	var ch ChallengeDetail
	if err := c.do(http.MethodGet, "/challenges/"+pathEscape(id), nil, &ch); err != nil {
		return nil, err
	}
	return &ch, nil
}

// SubmissionResult is the platform's verdict for one submitted answer.
type SubmissionResult struct {
	ID                  string `json:"id"`
	Correct             bool   `json:"correct"`
	QuestionID          *string `json:"question_id"`
	ChallengeFullySolved bool  `json:"challenge_fully_solved"`
	AttemptsRemaining   *int   `json:"attempts_remaining"`
	Message             string `json:"message,omitempty"`
}

// Submit posts one answer for this run's instance (CTFY-0001 §2, instance mode).
func (c *Client) Submit(instanceID, questionID, answer string) (*SubmissionResult, error) {
	body := map[string]any{"instance_id": instanceID, "answer": answer}
	if questionID != "" {
		body["question_id"] = questionID
	}
	var res SubmissionResult
	if err := c.do(http.MethodPost, "/submissions", body, &res); err != nil {
		return nil, err
	}
	return &res, nil
}

// SubmitQA posts one answer for an instance-free challenge (qa mode).
func (c *Client) SubmitQA(challengeID, competitionID, questionID, answer string) (*SubmissionResult, error) {
	if questionID == "" {
		questionID = "answer"
	}
	body := map[string]any{
		"challenge_id":   challengeID,
		"competition_id": competitionID,
		"question_id":    questionID,
		"answer":         answer,
	}
	var res SubmissionResult
	if err := c.do(http.MethodPost, "/qa/submissions", body, &res); err != nil {
		return nil, err
	}
	return &res, nil
}

// VerifyAnswer asks the grader without recording a submission (oracle mode).
// Useful to reject a candidate cheaply before spending an attempt.
func (c *Client) VerifyAnswer(instanceID, questionID, answer string) (bool, error) {
	body := map[string]any{"answer": answer}
	if questionID != "" {
		body["question_id"] = questionID
	}
	var res struct {
		Correct bool `json:"correct"`
	}
	if err := c.do(http.MethodPost, "/instances/"+pathEscape(instanceID)+"/verify-answer", body, &res); err != nil {
		return false, err
	}
	return res.Correct, nil
}

// SaveReport saves the whole engagement report (last write before the window
// closes is graded; CTFY-0001 §2 engagement mode).
func (c *Client) SaveReport(competitionID, challengeID, content string) error {
	path := "/competitions/" + pathEscape(competitionID) + "/challenges/" + pathEscape(challengeID) + "/report"
	return c.do(http.MethodPut, path, map[string]any{"content": content}, nil)
}

// AWDTarget is one gamebox in a match; Own marks this team's box.
type AWDTarget struct {
	Address  string `json:"address"`
	URL      string `json:"url"`
	Own      bool   `json:"is_own"`
	TeamName string `json:"team_name"`
}

// AWDTargets lists the gameboxes in scope for a match.
func (c *Client) AWDTargets(competitionID string) ([]AWDTarget, error) {
	var out struct {
		Items []AWDTarget `json:"items"`
	}
	err := c.do(http.MethodGet, "/competitions/"+pathEscape(competitionID)+"/awd/targets", nil, &out)
	return out.Items, err
}

// SubmitAWDFlag banks a flag captured from another team's box.
func (c *Client) SubmitAWDFlag(competitionID, flag string) error {
	return c.do(http.MethodPost, "/competitions/"+pathEscape(competitionID)+"/awd/submissions",
		map[string]any{"flag": flag}, nil)
}

// SortedQuestionIDs is a small helper for stable logging/ordering.
func SortedQuestionIDs(qs []Question) []string {
	ids := make([]string, 0, len(qs))
	for _, q := range qs {
		ids = append(ids, q.ID)
	}
	sort.Strings(ids)
	return ids
}

// pathEscape escapes a single path segment (ids are opaque strings).
func pathEscape(s string) string {
	var b strings.Builder
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9',
			r == '-', r == '_', r == '.', r == '~':
			b.WriteRune(r)
		default:
			b.WriteString(fmt.Sprintf("%%%02X", r))
		}
	}
	return b.String()
}

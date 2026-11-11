package ctfy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fakePlatform stands in for the CTFY platform: it answers /me, /submissions
// and /challenges/{id} just enough to exercise the client end to end.
func fakePlatform(t *testing.T) (*httptest.Server, *int) {
	t.Helper()
	var submits int
	mux := http.NewServeMux()
	mux.HandleFunc("/api/v1/me", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer tok" {
			w.WriteHeader(401)
			_, _ = w.Write([]byte(`{"code":"unauthorized"}`))
			return
		}
		_, _ = w.Write([]byte(`{"id":"u1"}`))
	})
	mux.HandleFunc("/api/v1/submissions", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(405)
			return
		}
		submits++
		var body struct {
			InstanceID string `json:"instance_id"`
			QuestionID string `json:"question_id"`
			Answer     string `json:"answer"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		correct := body.Answer == "flag{right}"
		_, _ = w.Write([]byte(`{"id":"s1","correct":` + boolStr(correct) +
			`,"question_id":` + strJSON(body.QuestionID) + `,"challenge_fully_solved":` + boolStr(correct) + `}`))
	})
	mux.HandleFunc("/api/v1/challenges/CH-1", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"CH-1","name":"Demo","questions":[
			{"id":"flag","prompt":"first"},
			{"id":"stage2","prompt":"now visible","requires":["flag"]}]}`))
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv, &submits
}

func boolStr(b bool) string {
	if b {
		return "true"
	}
	return "false"
}

func strJSON(s string) string { b, _ := json.Marshal(s); return string(b) }

func TestSubmitRecordsCorrectAnswer(t *testing.T) {
	srv, submits := fakePlatform(t)
	c := NewClient(srv.URL, "tok")

	res, err := c.Submit("inst-1", "flag", "flag{right}")
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	if !res.Correct {
		t.Error("expected correct=true")
	}
	if *submits != 1 {
		t.Errorf("submissions = %d, want 1", *submits)
	}
}

func TestSubmitWrongAnswerIsReported(t *testing.T) {
	srv, _ := fakePlatform(t)
	c := NewClient(srv.URL, "tok")
	res, err := c.Submit("inst-1", "flag", "flag{nope}")
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	if res.Correct {
		t.Error("expected correct=false")
	}
}

func TestClientWithoutCredentialsFailsExplicitly(t *testing.T) {
	c := NewClient("", "")
	if c.Available() {
		t.Fatal("a client with no URL or token must not be Available")
	}
	if _, err := c.Submit("i", "flag", "x"); err == nil {
		t.Error("expected an explicit error, not a silent no-op")
	}
}

func TestGetChallengeSeesNewlyUnlockedQuestions(t *testing.T) {
	srv, _ := fakePlatform(t)
	c := NewClient(srv.URL, "tok")
	ch, err := c.GetChallenge("CH-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(ch.Questions) != 2 || ch.Questions[1].ID != "stage2" {
		t.Errorf("questions = %+v", ch.Questions)
	}
	if got := ch.Questions[1].Requires; len(got) != 1 || got[0] != "flag" {
		t.Errorf("requires = %v", got)
	}
}

func TestAPIErrorCarriesCode(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(403)
		_, _ = w.Write([]byte(`{"code":"token_challenge_forbidden","message":"not this challenge"}`))
	}))
	defer srv.Close()
	c := NewClient(srv.URL, "tok")
	_, err := c.Submit("i", "flag", "x")
	if err == nil {
		t.Fatal("expected an error")
	}
	apiErr, ok := err.(*APIError)
	if !ok {
		t.Fatalf("error type = %T", err)
	}
	if apiErr.Code != "token_challenge_forbidden" || apiErr.Status != 403 {
		t.Errorf("APIError = %+v", apiErr)
	}
}

func TestReadTaskRejectsNewerSchema(t *testing.T) {
	dir := t.TempDir()
	_ = os.WriteFile(filepath.Join(dir, "task.json"), []byte(`{"schema_version":99,"run_mode":"instance"}`), 0o644)
	if _, err := ReadTask(dir); err == nil || !strings.Contains(err.Error(), "newer") {
		t.Errorf("err = %v, want a refusal for a newer schema", err)
	}
}

func TestReadTaskKeepsNullBudget(t *testing.T) {
	dir := t.TempDir()
	_ = os.WriteFile(filepath.Join(dir, "task.json"), []byte(`{"schema_version":3,"run_mode":"qa","max_turns":null}`), 0o644)
	task, err := ReadTask(dir)
	if err != nil {
		t.Fatal(err)
	}
	if task.MaxTurns != nil {
		t.Errorf("MaxTurns = %v, want nil (null is 'no budget', never 0)", *task.MaxTurns)
	}
}

func TestLoadEnvReadsPlatformContract(t *testing.T) {
	t.Setenv("CTFY_PLATFORM_URL", "https://ctfy.example/")
	t.Setenv("CTFY_TOKEN", " tok ")
	t.Setenv("CTFY_WORK_DIR", "")
	t.Setenv("LLM_FLAVOR", "OpenAI")
	t.Setenv("LLM_MODEL", "gpt-x")
	t.Setenv("LLM_API_KEY", "key")
	t.Setenv("LLM_BASE_URL", "https://gw.test/")
	t.Setenv("CTFY_EVAL_DRY_RUN", "")

	e := LoadEnv()
	if e.PlatformURL != "https://ctfy.example" {
		t.Errorf("PlatformURL = %q", e.PlatformURL)
	}
	if e.Token != "tok" {
		t.Errorf("Token = %q", e.Token)
	}
	if e.WorkDir != DefaultWorkDir {
		t.Errorf("WorkDir = %q, want the documented default", e.WorkDir)
	}
	if e.LLMFlavor != "openai" {
		t.Errorf("LLMFlavor = %q, want lower-cased", e.LLMFlavor)
	}
	if e.LLMBaseURL != "https://gw.test" {
		t.Errorf("LLMBaseURL = %q", e.LLMBaseURL)
	}
}

func TestLoadEnvDryRunFlag(t *testing.T) {
	t.Setenv("CTFY_EVAL_DRY_RUN", "1")
	if !LoadEnv().DryRun {
		t.Error("CTFY_EVAL_DRY_RUN=1 must set DryRun")
	}
}

package main

import (
	"context"
	"encoding/json"
	"iter"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Autumn-27/artex/ctfy"
	"github.com/Autumn-27/norma/llm"
)

// scriptedProvider answers each model call from a fixed list. Each step is a
// full assistant message: text and/or one tool call. When the steps run out it
// returns a plain end-of-turn message so the loop terminates.
type scriptedProvider struct {
	steps []llm.Message
	calls int
}

func (p *scriptedProvider) next() llm.Message {
	if p.calls < len(p.steps) {
		m := p.steps[p.calls]
		p.calls++
		return m
	}
	p.calls++
	return llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{llm.TextBlock("done")}}
}

func (p *scriptedProvider) Complete(ctx context.Context, req llm.CompletionRequest) (llm.Message, string, llm.Usage, error) {
	return p.next(), "end_turn", llm.Usage{InputTokens: 10, OutputTokens: 5}, nil
}

func (p *scriptedProvider) Stream(ctx context.Context, req llm.CompletionRequest) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		m := p.next()
		for _, b := range m.Content {
			switch b.Type {
			case llm.BlockText:
				if !yield(llm.StreamEvent{Type: llm.SETextDelta, Text: b.Text}, nil) {
					return
				}
			case llm.BlockToolUse:
				if !yield(llm.StreamEvent{Type: llm.SEToolUseStart, ToolID: b.ID, ToolName: b.Name}, nil) {
					return
				}
				if !yield(llm.StreamEvent{Type: llm.SEToolInputJSON, Text: string(b.Input)}, nil) {
					return
				}
			}
		}
		yield(llm.StreamEvent{Type: llm.SEMessageDelta, StopReason: "end_turn", Usage: llm.Usage{InputTokens: 10, OutputTokens: 5}}, nil)
	}
}

// TestRunnerSubmitsAndWritesResult drives the whole runner with a fake model:
// the model reads the briefing (via list_questions), submits a flag, and stops.
// Because this run has no platform credentials, doSubmit falls back to handing
// the answer over in result.json — which is exactly the path we want to prove.
func TestRunnerSubmitsAndWritesResult(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "task.json"), `{
	  "schema_version": 3,
	  "run_mode": "instance",
	  "challenge": {"id": "DEMO-001", "name": "Demo", "description": "find it"},
	  "competition_id": "demo",
	  "questions": [{"id": "flag", "prompt": "the secret"}],
	  "gated_questions": 0,
	  "result": {"path": "result.json"},
	  "target_endpoints": [{"label":"web","type":"http","url":"http://127.0.0.1:1"}]
	}`)
	writeFile(t, filepath.Join(dir, "README.md"), "# brief\n")

	task, err := ctfy.ReadTask(dir)
	if err != nil {
		t.Fatalf("ReadTask: %v", err)
	}
	env := ctfy.Env{WorkDir: dir, LLMModel: "fake", LLMAPIKey: "fake-key", LLMFlavor: "anthropic"}

	r := newRunner(dir, task, env, "solver")
	r.newProvider = func() (llm.Provider, string, string, error) {
		return &scriptedProvider{steps: []llm.Message{
			{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
				{Type: llm.BlockToolUse, ID: "t1", Name: "submit_answer",
					Input: json.RawMessage(`{"question_id":"flag","answer":"flag{demo_ok}"}`)},
			}},
			{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
				llm.TextBlock("submitted"),
			}},
		}}, "fake", "fake", nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	res := r.execute(ctx, 6, 30*time.Second)

	if res.StopReason != "done" {
		t.Errorf("stop_reason = %q, want done (writeup=%q)", res.StopReason, res.Writeup)
	}
	if got := res.Answers["flag"]; got != "flag{demo_ok}" {
		t.Errorf("answers[flag] = %q, want the submitted value", got)
	}
	if res.ToolCallsMade == 0 {
		t.Error("expected at least one tool call to be counted")
	}
	if err := ctfy.WriteResult(dir, res); err != nil {
		t.Fatalf("WriteResult: %v", err)
	}
	b, err := os.ReadFile(filepath.Join(dir, "result.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), "flag{demo_ok}") {
		t.Errorf("result.json missing the answer:\n%s", b)
	}
	if !strings.Contains(string(b), `"schema_version": 3`) {
		t.Errorf("result.json missing schema_version 3:\n%s", b)
	}
}

// TestDryRunWritesResult pins the conformance behaviour from §3.1: a clean exit
// with a minimal result file.
func TestDryRunWritesResult(t *testing.T) {
	dir := t.TempDir()
	if code := dryRunExit(dir, ctfy.Env{WorkDir: dir}); code != 0 {
		t.Errorf("dryRunExit = %d, want 0", code)
	}
	b, err := os.ReadFile(filepath.Join(dir, "result.json"))
	if err != nil {
		t.Fatalf("result.json: %v", err)
	}
	var res ctfy.Result
	if err := json.Unmarshal(b, &res); err != nil {
		t.Fatalf("parse result.json: %v", err)
	}
	if res.StopReason != "dry_run" || res.SchemaVersion != ctfy.SchemaVersion {
		t.Errorf("unexpected dry-run result: %+v", res)
	}
}

// TestUnfundedRunFailsLoudly covers §1.1: an empty key must stop the run rather
// than quietly spending another account.
func TestUnfundedRunFailsLoudly(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "task.json"), `{"schema_version":3,"run_mode":"instance","challenge":{"id":"X"},"questions":[],"result":{"path":"result.json"}}`)
	task, err := ctfy.ReadTask(dir)
	if err != nil {
		t.Fatal(err)
	}
	r := newRunner(dir, task, ctfy.Env{WorkDir: dir}, "solver")
	res := r.execute(context.Background(), 3, time.Minute)
	if res.StopReason != "unfunded" {
		t.Errorf("stop_reason = %q, want unfunded", res.StopReason)
	}
}

// TestProviderConfigHonoursPlatformContract checks the LLM_* mapping: the key
// and base URL travel together and the model id is required.
func TestProviderConfigHonoursPlatformContract(t *testing.T) {
	dir := t.TempDir()
	task := &ctfy.Task{SchemaVersion: 3, RunMode: ctfy.ModeInstance}

	r := newRunner(dir, task, ctfy.Env{LLMFlavor: "anthropic", LLMModel: "claude-x", LLMAPIKey: "k", LLMBaseURL: "https://gw.test/"}, "solver")
	cfg, err := r.providerConfig()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.BaseURL != "https://gw.test" || cfg.APIKey != "k" || cfg.Model != "claude-x" {
		t.Errorf("cfg = %+v", cfg)
	}
	if cfg.Format != llm.FormatAnthropic {
		t.Errorf("format = %v, want anthropic", cfg.Format)
	}

	r2 := newRunner(dir, task, ctfy.Env{LLMFlavor: "openai", LLMModel: "gpt-x", LLMAPIKey: "k"}, "solver")
	cfg2, err := r2.providerConfig()
	if err != nil {
		t.Fatal(err)
	}
	if cfg2.Format != llm.FormatOpenAI {
		t.Errorf("format = %v, want openai", cfg2.Format)
	}

	r3 := newRunner(dir, task, ctfy.Env{LLMAPIKey: "k"}, "solver")
	if _, err := r3.providerConfig(); err == nil {
		t.Error("expected an error when LLM_MODEL is empty")
	}
}

// TestDefaultQuestionID pins the fallbacks that keep a one-question challenge
// from needing an explicit id on every call.
func TestDefaultQuestionID(t *testing.T) {
	cases := []struct {
		task *ctfy.Task
		want string
	}{
		{&ctfy.Task{Questions: []ctfy.Question{{ID: "only"}}}, "only"},
		{&ctfy.Task{Questions: []ctfy.Question{{ID: "a"}, {ID: "flag", Prompt: "x"}}}, "flag"},
		{&ctfy.Task{RunMode: ctfy.ModeQA}, "answer"},
		{&ctfy.Task{}, "flag"},
	}
	for _, c := range cases {
		if got := defaultQuestionID(c.task); got != c.want {
			t.Errorf("defaultQuestionID(%+v) = %q, want %q", c.task.Questions, got, c.want)
		}
	}
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

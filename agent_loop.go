package main

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
)

// ToolHandler processes a single tool call and returns a result string.
// Returning an error causes the result to be surfaced as an error string
// in the session rather than failing the agent loop.
type ToolHandler func(input json.RawMessage) (string, error)

// Tool pairs a generic tool definition with its handler function.
type Tool struct {
	Definition ToolDefinition
	Handler    ToolHandler
}

// InitSession creates a session primed with a system prompt and an initial
// user message (guide section 1).
func InitSession(systemPrompt, userPrompt string) Session {
	s := Session{}
	s.Add(SystemMessage{systemPrompt}, UserMessage{userPrompt})
	return s
}

// ExecuteToolCalls runs all tool handlers concurrently (guide section 4) and
// returns a ToolResultMessage for each call.  Handler errors are captured as
// result strings so the agent loop can continue uninterrupted.
func ExecuteToolCalls(calls []ToolCallMessage, handlers map[string]ToolHandler) []Message {
	results := make([]Message, len(calls))
	var wg sync.WaitGroup
	for i, call := range calls {
		wg.Add(1)
		go func(i int, call ToolCallMessage) {
			defer wg.Done()
			handler, ok := handlers[call.Name]
			var output string
			if !ok {
				output = fmt.Sprintf("Error: unknown tool %q", call.Name)
			} else {
				out, err := handler(call.Input)
				if err != nil {
					output = "Error: " + err.Error()
				} else {
					output = out
				}
			}
			results[i] = ToolResultMessage{ID: call.ID, Output: output}
		}(i, call)
	}
	wg.Wait()
	return results
}

// AgentLoopOption configures a single AgentLoop call.
type AgentLoopOption func(*agentLoopConfig)

// LogFunc is called for each new message as it is produced during the loop.
type LogFunc func(Message)

// CompactFunc reduces a session before each model invocation to limit token
// bloat from accumulated history.  Pass nil via WithCompactor to disable.
type CompactFunc func(Session) Session

type agentLoopConfig struct {
	maxIterations int
	logFunc       LogFunc
	compactFunc   CompactFunc
}

// WithMaxIterations sets the maximum number of model invocations before the
// loop is terminated with an error.
func WithMaxIterations(n int) AgentLoopOption {
	return func(c *agentLoopConfig) { c.maxIterations = n }
}

// WithLogger sets a function that is called for each new message as it is
// produced — model responses (text, thinking, tool calls) and tool results.
func WithLogger(fn LogFunc) AgentLoopOption {
	return func(c *agentLoopConfig) { c.logFunc = fn }
}

// WithCompactor overrides the session compaction function.  Pass nil to
// disable compaction entirely.
func WithCompactor(fn CompactFunc) AgentLoopOption {
	return func(c *agentLoopConfig) { c.compactFunc = fn }
}

// defaultCompactor returns a CompactFunc that truncates ThinkingMessage,
// ToolCallMessage, and ToolResultMessage content once at least two assistant
// responses have appeared after them in the session.  A per-call index set
// prevents re-processing already-compacted messages on subsequent invocations.
func defaultCompactor() CompactFunc {
	const (
		assistantThreshold = 2   // assistant turns that must follow before compacting
		prefixLen          = 200 // bytes to keep from each compacted message
	)
	compacted := make(map[int]bool)

	return func(s Session) Session {
		assistantsSeen := 0
		for i := len(s.Messages) - 1; i >= 0; i-- {
			switch m := s.Messages[i].(type) {
			case AssistantMessage:
				assistantsSeen++
			case ThinkingMessage:
				if compacted[i] || assistantsSeen < assistantThreshold {
					continue
				}
				if len(m.Content) > prefixLen {
					s.Messages[i] = ThinkingMessage{Content: m.Content[:prefixLen] + "…"}
				}
				compacted[i] = true
			case ToolCallMessage:
				if compacted[i] || assistantsSeen < assistantThreshold {
					continue
				}
				raw := string(m.Input)
				if len(raw) > prefixLen {
					truncated, _ := json.Marshal(raw[:prefixLen] + "…")
					m.Input = truncated
					s.Messages[i] = m
				}
				compacted[i] = true
			case ToolResultMessage:
				if compacted[i] || assistantsSeen < assistantThreshold {
					continue
				}
				if len(m.Output) > prefixLen {
					m.Output = m.Output[:prefixLen] + "…"
					s.Messages[i] = m
				}
				compacted[i] = true
			}
		}
		return s
	}
}

// AgentLoop drives the model in a loop until it produces a response with no
// tool calls (guide section 5).  The updated session is returned.
//
// invokeModel is the model invocation function (e.g. InvokeClaude()).
// tools provides both the definitions passed to invokeModel and the handler
// functions used to execute them.
func AgentLoop(ctx context.Context, invokeModel InvokeModelFunc, tools []Tool, session Session, opts ...AgentLoopOption) (Session, error) {
	cfg := &agentLoopConfig{maxIterations: 30, compactFunc: defaultCompactor()}
	for _, o := range opts {
		o(cfg)
	}

	// Build a definition slice (for the API) and a handler map (for dispatch).
	defs := make([]ToolDefinition, len(tools))
	handlers := make(map[string]ToolHandler, len(tools))
	for i, t := range tools {
		defs[i] = t.Definition
		handlers[t.Definition.Name] = t.Handler
	}

	for i := range cfg.maxIterations {
		if cfg.compactFunc != nil {
			session = cfg.compactFunc(session)
		}

		newMsgs, err := invokeModel(ctx, defs, session)
		if err != nil {
			return session, err
		}
		session.Add(newMsgs...)
		if cfg.logFunc != nil {
			for _, m := range newMsgs {
				cfg.logFunc(m)
			}
		}

		// Collect tool calls from this turn.
		var toolCalls []ToolCallMessage
		for _, m := range newMsgs {
			if tc, ok := m.(ToolCallMessage); ok {
				toolCalls = append(toolCalls, tc)
			}
		}

		// No tool calls means the model is done.
		if len(toolCalls) == 0 {
			break
		}

		if i == cfg.maxIterations-1 {
			return session, fmt.Errorf("agent loop reached maximum iterations (%d)", cfg.maxIterations)
		}

		results := ExecuteToolCalls(toolCalls, handlers)
		session.Add(results...)
		if cfg.logFunc != nil {
			for _, m := range results {
				cfg.logFunc(m)
			}
		}
	}

	return session, nil
}

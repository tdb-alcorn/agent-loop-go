package main

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	anthropic "github.com/anthropics/anthropic-sdk-go"
)

// ToolHandler processes a single tool call and returns a result string.
// Returning an error causes the result to be surfaced as an error string
// in the session rather than failing the agent loop.
type ToolHandler func(input json.RawMessage) (string, error)

// Tool pairs an Anthropic tool definition with its handler function.
// Name must match the name embedded in Definition.
type Tool struct {
	Name       string
	Definition anthropic.ToolUnionParam
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

// AgentLoop drives the model in a loop until it produces a response with no
// tool calls (guide section 5).  The updated session is returned.
//
// tools provides both the definitions passed to the model and the handler
// functions used to execute them.  Additional opts are forwarded to
// InvokeModel on every iteration (e.g. WithMaxTokens, WithThinking).
func AgentLoop(ctx context.Context, client *Client, tools []Tool, session Session, opts ...Option) (Session, error) {
	// Build a definition slice (for the API) and a handler map (for dispatch).
	defs := make([]anthropic.ToolUnionParam, len(tools))
	handlers := make(map[string]ToolHandler, len(tools))
	for i, t := range tools {
		defs[i] = t.Definition
		handlers[t.Name] = t.Handler
	}

	// Prepend tool definitions to every InvokeModel call.
	callOpts := make([]Option, 0, len(opts)+1)
	if len(defs) > 0 {
		callOpts = append(callOpts, WithTools(defs...))
	}
	callOpts = append(callOpts, opts...)

	for {
		newMsgs, err := InvokeModel(ctx, client, session, callOpts...)
		if err != nil {
			return session, err
		}
		session.Add(newMsgs...)

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

		results := ExecuteToolCalls(toolCalls, handlers)
		session.Add(results...)
	}

	return session, nil
}

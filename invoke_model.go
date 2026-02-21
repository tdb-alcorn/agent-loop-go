package main

import (
	"context"

	anthropic "github.com/anthropics/anthropic-sdk-go"
)

// InvokeModel converts session into Anthropic API params, calls the model,
// and returns the new messages produced by the response.
//
// Conversion rules:
//   - SystemMessage        → params.System (TextBlockParam)
//   - UserMessage          → user turn, text block
//   - AssistantMessage     → assistant turn, text block
//   - ThinkingMessage      → skipped (no API signature; kept in session for display only)
//   - ToolCallMessage      → assistant turn, tool_use block
//   - ToolResultMessage    → user turn, tool_result block
//
// Consecutive messages of the same role are merged into a single turn.
func InvokeModel(ctx context.Context, client *Client, session Session, opts ...Option) ([]Message, error) {
	system, messages := buildParams(session)

	cfg := &completeConfig{
		model:     client.model,
		maxTokens: 4096,
	}
	for _, o := range opts {
		o(cfg)
	}

	params := anthropic.MessageNewParams{
		Model:     cfg.model,
		MaxTokens: cfg.maxTokens,
		Messages:  messages,
	}
	if len(system) > 0 {
		params.System = system
	}
	if cfg.thinking != nil {
		params.Thinking = anthropic.ThinkingConfigParamOfEnabled(*cfg.thinking)
	}
	if len(cfg.tools) > 0 {
		params.Tools = toolDefsToParams(cfg.tools)
	}

	resp, err := client.api.Messages.New(ctx, params)
	if err != nil {
		return nil, err
	}
	return responseToMessages(resp), nil
}

// buildParams converts a Session into the system blocks and message turns
// expected by the Anthropic API.
func buildParams(session Session) ([]anthropic.TextBlockParam, []anthropic.MessageParam) {
	var system []anthropic.TextBlockParam
	var turns []anthropic.MessageParam

	for _, msg := range session.Messages {
		if sm, ok := msg.(SystemMessage); ok {
			system = append(system, anthropic.TextBlockParam{Text: sm.Content})
			continue
		}

		role, block, ok := toBlock(msg)
		if !ok {
			continue // ThinkingMessage and unknowns are skipped
		}

		// Merge into the last turn if same role, otherwise start a new one.
		if len(turns) > 0 && turns[len(turns)-1].Role == role {
			turns[len(turns)-1].Content = append(turns[len(turns)-1].Content, block)
		} else {
			turns = append(turns, anthropic.MessageParam{
				Role:    role,
				Content: []anthropic.ContentBlockParamUnion{block},
			})
		}
	}

	return system, turns
}

// toBlock converts a session Message to an API role and content block.
// Returns ok=false for messages that should be omitted from the API request.
func toBlock(msg Message) (anthropic.MessageParamRole, anthropic.ContentBlockParamUnion, bool) {
	switch m := msg.(type) {
	case UserMessage:
		return anthropic.MessageParamRoleUser, anthropic.NewTextBlock(m.Content), true
	case AssistantMessage:
		return anthropic.MessageParamRoleAssistant, anthropic.NewTextBlock(m.Content), true
	case ToolCallMessage:
		return anthropic.MessageParamRoleAssistant, anthropic.NewToolUseBlock(m.ID, m.Input, m.Name), true
	case ToolResultMessage:
		return anthropic.MessageParamRoleUser, anthropic.NewToolResultBlock(m.ID, m.Output, false), true
	default:
		// SystemMessage is handled before this call; ThinkingMessage is skipped.
		return "", anthropic.ContentBlockParamUnion{}, false
	}
}

// toolDefsToParams converts a slice of generic ToolDefinitions to Anthropic API params.
func toolDefsToParams(defs []ToolDefinition) []anthropic.ToolUnionParam {
	params := make([]anthropic.ToolUnionParam, len(defs))
	for i, def := range defs {
		t := anthropic.ToolUnionParamOfTool(
			anthropic.ToolInputSchemaParam{
				Properties: def.InputSchema.Properties,
				Required:   def.InputSchema.Required,
			},
			def.Name,
		)
		if def.Description != "" {
			t.OfTool.Description = anthropic.String(def.Description)
		}
		params[i] = t
	}
	return params
}

// responseToMessages converts an Anthropic API response into session Messages.
func responseToMessages(resp *anthropic.Message) []Message {
	var out []Message
	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			out = append(out, AssistantMessage{block.AsText().Text})
		case "thinking":
			out = append(out, ThinkingMessage{block.AsThinking().Thinking})
		case "tool_use":
			tu := block.AsToolUse()
			out = append(out, ToolCallMessage{ID: tu.ID, Name: tu.Name, Input: tu.Input})
		}
	}
	return out
}

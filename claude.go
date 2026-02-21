package agentloop

import (
	"context"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// Claude wraps the Anthropic API with sensible defaults.
type Claude struct {
	api   anthropic.Client
	model anthropic.Model
}

// NewClaude creates a Claude client using ANTHROPIC_API_KEY from the environment.
func NewClaude(opts ...option.RequestOption) *Claude {
	return &Claude{
		api:   anthropic.NewClient(opts...),
		model: anthropic.ModelClaudeSonnet4_6,
	}
}

// completeConfig holds per-request options built by Option functions.
type completeConfig struct {
	model     anthropic.Model
	maxTokens int64
	system    string
	thinking  *int64 // budget tokens; nil = disabled
	tools     []ToolDefinition
}

// Option configures a single Complete call.
type Option func(*completeConfig)

// WithModel overrides the model for this request.
func WithModel(m anthropic.Model) Option {
	return func(c *completeConfig) { c.model = m }
}

// WithMaxTokens sets the maximum tokens to generate.
func WithMaxTokens(n int64) Option {
	return func(c *completeConfig) { c.maxTokens = n }
}

// WithSystem sets a system prompt for this request.
func WithSystem(s string) Option {
	return func(c *completeConfig) { c.system = s }
}

// WithThinking enables extended thinking with the given token budget (min 1024).
// The budget counts toward max_tokens, so set max_tokens accordingly.
func WithThinking(budgetTokens int64) Option {
	return func(c *completeConfig) { c.thinking = &budgetTokens }
}

// WithTools provides tool definitions the model may call.
func WithTools(tools ...ToolDefinition) Option {
	return func(c *completeConfig) { c.tools = tools }
}

// Complete sends a single user message and returns the model's response.
func (c *Claude) Complete(ctx context.Context, prompt string, opts ...Option) (*anthropic.Message, error) {
	cfg := &completeConfig{
		model:     c.model,
		maxTokens: 1024,
	}
	for _, o := range opts {
		o(cfg)
	}

	params := anthropic.MessageNewParams{
		Model:     cfg.model,
		MaxTokens: cfg.maxTokens,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(prompt)),
		},
	}
	if cfg.system != "" {
		params.System = []anthropic.TextBlockParam{{Text: cfg.system}}
	}
	if cfg.thinking != nil {
		params.Thinking = anthropic.ThinkingConfigParamOfEnabled(*cfg.thinking)
	}
	if len(cfg.tools) > 0 {
		params.Tools = toolDefsToParams(cfg.tools)
	}

	return c.api.Messages.New(ctx, params)
}

// TextContent returns the concatenated text from a message's content blocks.
func TextContent(msg *anthropic.Message) string {
	var out string
	for _, block := range msg.Content {
		if block.Type == "text" {
			out += block.AsText().Text
		}
	}
	return out
}

// ToolUseBlocks returns all tool_use blocks from a message.
func ToolUseBlocks(msg *anthropic.Message) []anthropic.ToolUseBlock {
	var blocks []anthropic.ToolUseBlock
	for _, block := range msg.Content {
		if block.Type == "tool_use" {
			blocks = append(blocks, block.AsToolUse())
		}
	}
	return blocks
}

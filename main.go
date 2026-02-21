package main

import (
	"fmt"
	"context"
)

func invoke_model() {
	client := NewClient()
	ctx := context.Background()

	msg, err := client.Complete(ctx, `Say exactly: "Hello, World!"`)
	if err != nil {
		fmt.Println(err)
	}

    fmt.Println(TextContent(msg))
	fmt.Println(msg.StopReason)
	fmt.Println(msg)
}

func main() {
	LoadDotEnv(".env")

	invoke_model()
}

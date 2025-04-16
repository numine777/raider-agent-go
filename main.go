package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/ollama/ollama/api"
)

const MODEL_NAME = "qwen2.5-coder:32b-instruct-q8_0"
const OLLAMA_HOST = "localhost:11435"

func main() {
	os.Setenv("OLLAMA_HOST", OLLAMA_HOST)
	client, err := api.ClientFromEnvironment()
	if err != nil {
		fmt.Printf("Error: %s\n", err.Error())
	}

	scanner := bufio.NewScanner(os.Stdin)
	getUserMessage := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}
		return scanner.Text(), true
	}

	agent := NewAgent(client, getUserMessage)
	err = agent.Run(context.TODO())
	if err != nil {
		fmt.Printf("Error: %s\n", err.Error())
	}
}

func NewAgent(client *api.Client, getUserMessage func() (string, bool)) *Agent {
	return &Agent{
		client:         client,
		getUserMessage: getUserMessage,
	}
}

type Agent struct {
	client         *api.Client
	getUserMessage func() (string, bool)
}

func (a *Agent) Run(ctx context.Context) error {
	messages := []api.Message{}
	conversation := &api.ChatRequest{
		Model:    MODEL_NAME,
		Messages: messages,
	}

	fmt.Println("Chat with Ollama (use 'ctrl-c' to quit)")

	for {
		fmt.Print("\u001b[94mYou\u001b[0m: ")
		userInput, ok := a.getUserMessage()
		if !ok {
			break
		}

		userMessage := api.Message{
			Role:    "user",
			Content: userInput,
		}
		messages = append(messages, userMessage)
		conversation.Messages = messages

		fmt.Print("\u001b[93mOllama\u001b[0m: ")
		responseTokens := []string{}
		var responder string
		handler := func(chat api.ChatResponse) error {
			message := chat.Message
			fmt.Print(message.Content)
			responseTokens = append(responseTokens, message.Content)
			responder = chat.Message.Role

			return nil
		}

		err := a.runInference(ctx, conversation, handler)
		if err != nil {
			return err
		}

		response := strings.Join(responseTokens, " ")
		responseMsg := api.Message{
			Content: response,
			Role: responder,
		}
		messages = append(messages, responseMsg)
		conversation.Messages = messages
		fmt.Print("\n")
	}

	return nil
}

func (a *Agent) runInference(ctx context.Context, conversation *api.ChatRequest, handler api.ChatResponseFunc) error {
	err := a.client.Chat(ctx, conversation, handler)

	return err
}

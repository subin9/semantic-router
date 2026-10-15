package testcases

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const sessionExportQuestion = "How do I rotate the deploy key for the staging cluster?"

type sessionExportTurn struct {
	RecordID         string   `json:"record_id"`
	TurnIndex        int      `json:"turn_index"`
	Decision         string   `json:"decision"`
	SelectedModel    string   `json:"selected_model"`
	NextTurnFeedback []string `json:"next_turn_feedback"`
}

type sessionExport struct {
	Version      string              `json:"version"`
	SessionCount int                 `json:"session_count"`
	Turns        []sessionExportTurn `json:"turns"`
}

func init() {
	pkgtestcases.Register("router-replay-session-export", pkgtestcases.TestCase{
		Description: "The sessions export reads two persisted turns of one session in order, joins the reask matched on the second turn to the first, and carries no captured text",
		Tags:        []string{"router-replay", "functional", "router-replay-api", "postgres"},
		Fn:          testRouterReplaySessionExport,
	})
}

func testRouterReplaySessionExport(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("open session: %w", err)
	}
	defer session.Close()
	apiSession, err := fixtures.OpenRouterAPISession(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("open Router management API session: %w", err)
	}
	defer apiSession.Close()

	sessionID := fmt.Sprintf("e2e_session_export_%d", time.Now().UnixNano())
	first := []fixtures.ChatMessage{{Role: "user", Content: sessionExportQuestion}}
	// The second turn asks the same question again, so the reask signal fires
	// on it and judges the first answer.
	second := append(slices.Clone(first),
		fixtures.ChatMessage{Role: "assistant", Content: "mock assistant reply for e2e"},
		fixtures.ChatMessage{Role: "user", Content: sessionExportQuestion},
	)
	for _, messages := range [][]fixtures.ChatMessage{first, second} {
		if err = postChatCompletionsWithReplayHeaders(ctx, session, postChatOpts{
			model:     "vllm-sr/session-export",
			sessionID: sessionID,
			userID:    "e2e-session-export-user",
			messages:  messages,
		}); err != nil {
			return err
		}
		time.Sleep(2 * time.Second)
	}

	raw, err := doRouterReplayManagementGET(ctx, apiSession,
		"/api/v1/observability/replays/sessions?session_id="+url.QueryEscape(sessionID))
	if err != nil {
		return fmt.Errorf("GET sessions export: %w", err)
	}
	if raw.StatusCode != http.StatusOK {
		return fmt.Errorf("sessions export status %d: %s", raw.StatusCode, string(raw.Body))
	}
	if strings.Contains(string(raw.Body), "deploy key") {
		return fmt.Errorf("sessions export carries captured text: %s", string(raw.Body))
	}
	var export sessionExport
	if err = raw.DecodeJSON(&export); err != nil {
		return fmt.Errorf("decode sessions export: %w", err)
	}
	if opts.Verbose {
		fmt.Printf("[Test] sessions export for %s: %+v\n", sessionID, export)
	}
	return requireSessionExportJoin(export)
}

func requireSessionExportJoin(export sessionExport) error {
	if export.SessionCount != 1 || len(export.Turns) != 2 {
		return fmt.Errorf("export has %d sessions and %d turns, want 1 and 2", export.SessionCount, len(export.Turns))
	}
	before, after := export.Turns[0], export.Turns[1]
	if before.TurnIndex != 0 || after.TurnIndex != 1 {
		return fmt.Errorf("turn indexes %d then %d, want 0 then 1", before.TurnIndex, after.TurnIndex)
	}
	// The decision proves the signal fired on the repeat, so a missing join is
	// the export's fault and not a signal that never matched.
	if after.Decision != "session_reask_decision" {
		return fmt.Errorf("repeat routed to %q, want session_reask_decision", after.Decision)
	}
	if !slices.Equal(before.NextTurnFeedback, []string{"repeated_question"}) {
		return fmt.Errorf("first turn feedback %v, want [repeated_question]", before.NextTurnFeedback)
	}
	if len(after.NextTurnFeedback) != 0 {
		return fmt.Errorf("repeat kept its own feedback %v", after.NextTurnFeedback)
	}
	return nil
}

package extproc

import (
	"strings"
	"testing"
	"time"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func sessionReplayRecord(id, session string, turn int) routerreplay.RoutingRecord {
	return routerreplay.RoutingRecord{
		ID:            id,
		Timestamp:     time.Date(2026, 10, 15, 9, 0, turn, 0, time.UTC),
		Recipe:        "coding",
		Decision:      "code",
		SessionID:     session,
		TurnIndex:     turn,
		SelectedModel: "small",
		RequestBody:   `{"messages":[{"role":"user","content":"private key ` + id + `"}]}`,
		ResponseBody:  `{"id":"resp-` + id + `"}`,
	}
}

// The route reaches the builder through the replay dispatcher, honours the
// replay filters, and returns no captured text even though the store holds it.
func TestRouterReplaySessionsExportsFilteredTurnsWithoutText(t *testing.T) {
	reply := sessionReplayRecord("a1", "s-a", 1)
	reply.Signals = store.Signal{UserFeedback: []string{"wrong_answer"}}
	router := newDatasetExportRouter(t,
		sessionReplayRecord("a0", "s-a", 0),
		reply,
		sessionReplayRecord("b0", "s-b", 0),
	)

	response := router.handleRouterReplayAPI("GET", "/api/v1/observability/replays/sessions?session_id=s-a")
	assertDatasetStatus(t, response, typev3.StatusCode_OK)
	raw := string(response.GetImmediateResponse().Body)
	if strings.Contains(raw, "private key") {
		t.Fatalf("session export leaked captured text: %s", raw)
	}
	body := decodeJSONBody(t, response.GetImmediateResponse().Body)
	assertStringField(t, body, "version", routerreplay.SessionDatasetVersion)
	assertIntField(t, body, "session_count", 1)
	turns, ok := body["turns"].([]interface{})
	if !ok || len(turns) != 2 {
		t.Fatalf("expected two turns of s-a, got %#v", body["turns"])
	}
	first, _ := turns[0].(map[string]interface{})
	if feedback, _ := first["next_turn_feedback"].([]interface{}); len(feedback) != 1 || feedback[0] != "wrong_answer" {
		t.Fatalf("turn 0 feedback %#v, want [wrong_answer]", first["next_turn_feedback"])
	}
}

func TestRouterReplaySessionsRejectsANonReadMethod(t *testing.T) {
	router := newDatasetExportRouter(t, sessionReplayRecord("a0", "s-a", 0))
	response := router.handleRouterReplayAPI("POST", "/api/v1/observability/replays/sessions")
	assertDatasetStatus(t, response, typev3.StatusCode_MethodNotAllowed)
}

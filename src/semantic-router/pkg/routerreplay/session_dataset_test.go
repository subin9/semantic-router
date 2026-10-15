package routerreplay

import (
	"reflect"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func sessionRecord(id, session string, turn int, previous, selected string) RoutingRecord {
	return RoutingRecord{
		ID:               id,
		Recipe:           "coding",
		SessionID:        session,
		TurnIndex:        turn,
		Timestamp:        time.Date(2026, 10, 15, 9, 0, turn, 0, time.UTC),
		SelectedModel:    selected,
		RouteDiagnostics: &store.RouteDiagnostics{PreviousModel: previous},
	}
}

func turnIDs(turns []SessionTurn) []string {
	ids := make([]string, len(turns))
	for i, turn := range turns {
		ids[i] = turn.RecordID
	}
	return ids
}

// Storage lists newest first and interleaves sessions, so the export has to
// restore turn order inside each session and a fixed order across sessions.
func TestBuildSessionDatasetOrdersTurnsWithinSessions(t *testing.T) {
	dataset := BuildSessionDataset([]RoutingRecord{
		sessionRecord("b1", "s-b", 1, "small", "small"),
		sessionRecord("a1", "s-a", 1, "small", "large"),
		sessionRecord("b0", "s-b", 0, "", "small"),
		sessionRecord("a0", "s-a", 0, "", "small"),
	})

	if got, want := turnIDs(dataset.Turns), []string{"a0", "a1", "b0", "b1"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("turn order %v, want %v", got, want)
	}
	if dataset.SessionCount != 2 || dataset.SwitchCount != 1 || dataset.RecordCount != 4 {
		t.Fatalf("counts sessions=%d switches=%d records=%d, want 2 1 4",
			dataset.SessionCount, dataset.SwitchCount, dataset.RecordCount)
	}
	if !dataset.Turns[1].Switched || dataset.Turns[3].Switched {
		t.Fatalf("switched flags %v %v, want true false", dataset.Turns[1].Switched, dataset.Turns[3].Switched)
	}
}

// Feedback signals read the user's reply, so they belong to the answer before
// it: the last row of the previous user turn, not the row they were matched on.
func TestBuildSessionDatasetJoinsFeedbackToThePreviousTurn(t *testing.T) {
	hop := sessionRecord("t0-hop", "s", 0, "", "small")
	hop.Timestamp = hop.Timestamp.Add(time.Millisecond)
	reply := sessionRecord("t1", "s", 1, "small", "small")
	reply.Signals = store.Signal{UserFeedback: []string{"wrong_answer"}, Reask: []string{"repeat"}}

	dataset := BuildSessionDataset([]RoutingRecord{reply, hop, sessionRecord("t0", "s", 0, "", "small")})

	turns := dataset.Turns
	if got, want := turnIDs(turns), []string{"t0", "t0-hop", "t1"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("turn order %v, want %v", got, want)
	}
	if turns[0].NextTurnFeedback != nil {
		t.Fatalf("a tool hop's first row took feedback %v", turns[0].NextTurnFeedback)
	}
	if got, want := turns[1].NextTurnFeedback, []string{"wrong_answer", "repeat"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("feedback %v, want %v", got, want)
	}
	if turns[2].NextTurnFeedback != nil {
		t.Fatalf("the reply row kept its own feedback %v", turns[2].NextTurnFeedback)
	}
}

// When the next user turn was not recorded, a later reply judges a different
// answer, so nothing joins across the gap.
func TestBuildSessionDatasetDoesNotJoinAcrossAMissingTurn(t *testing.T) {
	later := sessionRecord("t2", "s", 2, "small", "small")
	later.Signals = store.Signal{UserFeedback: []string{"satisfied"}}

	dataset := BuildSessionDataset([]RoutingRecord{sessionRecord("t0", "s", 0, "", "small"), later})

	if feedback := dataset.Turns[0].NextTurnFeedback; feedback != nil {
		t.Fatalf("turn 0 took feedback %v from turn 2", feedback)
	}
}

// Conversations inside one session are separate sequences (a subagent restarts
// its turn count), so turn order and the feedback join stay inside each one.
func TestBuildSessionDatasetKeepsConversationsApart(t *testing.T) {
	parent := sessionRecord("p0", "s", 0, "", "large")
	parent.ConversationID = "parent"
	child := sessionRecord("c1", "s", 1, "small", "small")
	child.ConversationID = "child"
	child.Signals = store.Signal{UserFeedback: []string{"satisfied"}}

	dataset := BuildSessionDataset([]RoutingRecord{parent, child})

	if dataset.SessionCount != 1 {
		t.Fatalf("session count %d, want 1", dataset.SessionCount)
	}
	if feedback := dataset.Turns[1].NextTurnFeedback; feedback != nil {
		t.Fatalf("feedback %v crossed from the child conversation", feedback)
	}
}

// The export is content-minimized: captured text never leaves with a turn,
// and records with no session are counted rather than dropped silently.
func TestBuildSessionDatasetCarriesNoTextAndCountsExclusions(t *testing.T) {
	record := sessionRecord("t0", "s", 0, "", "small")
	record.RequestBody = "secret prompt"
	record.ResponseBody = "secret answer"
	record.Learning = &store.LearningDiagnostics{Protection: &store.LearningProtectionDiagnostics{CacheWarmth: 0.8, CacheWarmthOK: true}}
	orphan := sessionRecord("x", "", 0, "", "small")

	dataset := BuildSessionDataset([]RoutingRecord{record, orphan})

	if got := dataset.Excluded[ExcludeSessionMissing]; got != 1 {
		t.Fatalf("excluded %v, want 1 %s", dataset.Excluded, ExcludeSessionMissing)
	}
	if len(dataset.Turns) != 1 || dataset.Turns[0].Protection == nil || dataset.Turns[0].Protection.CacheWarmth != 0.8 {
		t.Fatalf("protection evidence did not carry over: %+v", dataset.Turns)
	}
	if fields := reflect.TypeOf(SessionTurn{}); hasField(fields, "RequestBody") || hasField(fields, "ResponseBody") || hasField(fields, "Prompt") {
		t.Fatal("SessionTurn grew a captured-text field")
	}
}

func hasField(typ reflect.Type, name string) bool {
	_, ok := typ.FieldByName(name)
	return ok
}

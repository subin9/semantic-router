package routerreplay

import (
	"sort"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// SessionDatasetVersion identifies the session dataset shape. A change to the
// fields a turn carries or to how turns are joined changes this string, because
// an offline consumer built against one shape cannot read another.
const SessionDatasetVersion = "router-replay-sessions.v1"

// ExcludeSessionMissing counts records with no logical session. A turn outside
// a session has no previous model to switch from and no next turn to judge it.
const ExcludeSessionMissing = "session_id_missing"

// SessionDataset is the content-minimized, per-session view of Router Replay
// that offline learning reads: which model served each turn, whether the turn
// switched, what the stay or switch evidence was, and what came back after.
// It never carries prompt, response, or tool text.
type SessionDataset struct {
	Object       string         `json:"object"`
	Version      string         `json:"version"`
	RecordCount  int            `json:"record_count"`
	SessionCount int            `json:"session_count"`
	SwitchCount  int            `json:"switch_count"`
	Excluded     map[string]int `json:"excluded,omitempty"`
	Turns        []SessionTurn  `json:"turns"`
}

// SessionTurn is one recorded routing result inside a session. Tool hops of
// one user turn share TurnIndex and appear as separate rows in record order.
type SessionTurn struct {
	RecordID           string    `json:"record_id"`
	Recipe             string    `json:"recipe"`
	SessionID          string    `json:"session_id"`
	ConversationID     string    `json:"conversation_id,omitempty"`
	TurnIndex          int       `json:"turn_index"`
	Timestamp          time.Time `json:"timestamp"`
	Decision           string    `json:"decision,omitempty"`
	SelectionMethod    string    `json:"selection_method,omitempty"`
	PreviousModel      string    `json:"previous_model,omitempty"`
	SelectedModel      string    `json:"selected_model"`
	Switched           bool      `json:"switched"`
	LifecycleState     string    `json:"lifecycle_state,omitempty"`
	ResponseStatus     int       `json:"response_status,omitempty"`
	DurationMS         int64     `json:"duration_ms"`
	PromptTokens       *int      `json:"prompt_tokens,omitempty"`
	CachedPromptTokens *int      `json:"cached_prompt_tokens,omitempty"`
	CompletionTokens   *int      `json:"completion_tokens,omitempty"`
	ActualCost         *float64  `json:"actual_cost,omitempty"`
	// Protection is the stay or switch evidence Router Learning recorded:
	// cache warmth, tool-loop and hard-lock state, and per-candidate switch
	// advantage. It is absent when the decision ran without protection.
	Protection *store.LearningProtectionDiagnostics `json:"protection,omitempty"`
	Outcomes   []store.Outcome                      `json:"outcomes,omitempty"`
	// NextTurnFeedback holds the user_feedback and reask signals matched on the
	// next user turn of the same conversation. Those signals read the user's
	// reply, so they judge this turn's answer, not the turn they were recorded on.
	// Only the last row of a user turn carries them.
	NextTurnFeedback []string `json:"next_turn_feedback,omitempty"`
}

type sessionKey struct{ recipe, session, conversation string }

// BuildSessionDataset groups records by recipe, session and conversation,
// orders each group by turn, and joins every user turn with the feedback
// matched on the turn after it. Output order is fully determined by the
// records, so the same selection always exports the same rows.
func BuildSessionDataset(records []RoutingRecord) SessionDataset {
	dataset := SessionDataset{
		Object:      "router_replay.session_dataset",
		Version:     SessionDatasetVersion,
		RecordCount: len(records),
		Turns:       []SessionTurn{},
	}
	groups := map[sessionKey][]RoutingRecord{}
	for _, record := range records {
		if record.SessionID == "" {
			if dataset.Excluded == nil {
				dataset.Excluded = map[string]int{}
			}
			dataset.Excluded[ExcludeSessionMissing]++
			continue
		}
		key := sessionKey{record.Recipe, record.SessionID, record.ConversationID}
		groups[key] = append(groups[key], record)
	}

	keys := make([]sessionKey, 0, len(groups))
	sessions := map[[2]string]struct{}{}
	for key := range groups {
		keys = append(keys, key)
		sessions[[2]string{key.recipe, key.session}] = struct{}{}
	}
	dataset.SessionCount = len(sessions)
	sort.Slice(keys, func(i, j int) bool {
		a, b := keys[i], keys[j]
		if a.recipe != b.recipe {
			return a.recipe < b.recipe
		}
		if a.session != b.session {
			return a.session < b.session
		}
		return a.conversation < b.conversation
	})

	for _, key := range keys {
		turns := sessionTurns(groups[key])
		for _, turn := range turns {
			if turn.Switched {
				dataset.SwitchCount++
			}
		}
		dataset.Turns = append(dataset.Turns, turns...)
	}
	return dataset
}

func sessionTurns(records []RoutingRecord) []SessionTurn {
	sort.Slice(records, func(i, j int) bool {
		a, b := records[i], records[j]
		if a.TurnIndex != b.TurnIndex {
			return a.TurnIndex < b.TurnIndex
		}
		if !a.Timestamp.Equal(b.Timestamp) {
			return a.Timestamp.Before(b.Timestamp)
		}
		return a.ID < b.ID
	})
	turns := make([]SessionTurn, len(records))
	for i, record := range records {
		turns[i] = sessionTurn(record)
		// The first row of the next user turn carries the reply to this one. A
		// gap in turn indexes means that reply was not recorded, so nothing joins.
		if i+1 < len(records) && records[i+1].TurnIndex == record.TurnIndex+1 {
			next := records[i+1].Signals
			feedback := append(append([]string{}, next.UserFeedback...), next.Reask...)
			if len(feedback) > 0 {
				turns[i].NextTurnFeedback = feedback
			}
		}
	}
	return turns
}

func sessionTurn(record RoutingRecord) SessionTurn {
	turn := SessionTurn{
		RecordID:           record.ID,
		Recipe:             record.Recipe,
		SessionID:          record.SessionID,
		ConversationID:     record.ConversationID,
		TurnIndex:          record.TurnIndex,
		Timestamp:          record.Timestamp,
		Decision:           record.Decision,
		SelectionMethod:    record.SelectionMethod,
		SelectedModel:      record.SelectedModel,
		LifecycleState:     record.LifecycleState,
		ResponseStatus:     record.ResponseStatus,
		DurationMS:         record.DurationMS,
		PromptTokens:       record.PromptTokens,
		CachedPromptTokens: record.CachedPromptTokens,
		CompletionTokens:   record.CompletionTokens,
		ActualCost:         record.ActualCost,
		Outcomes:           record.Outcomes,
	}
	if record.RouteDiagnostics != nil {
		turn.PreviousModel = record.RouteDiagnostics.PreviousModel
	}
	turn.Switched = turn.PreviousModel != "" && turn.PreviousModel != turn.SelectedModel
	if record.Learning != nil {
		turn.Protection = record.Learning.Protection
	}
	return turn
}

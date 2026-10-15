package extproc

import (
	"errors"
	"net/url"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

const routerReplaySessionsPath = routerReplayAPIBasePath + "/sessions"

// handleRouterReplaySessionsAPI serves GET /api/v1/observability/replays/sessions.
// It exports the selected records as per-session turns with stay or switch
// evidence, outcomes, and the feedback each answer drew, for offline learning.
// It reads the same selection and cap as the dataset export, so an oversized
// selection is refused rather than cut mid-session.
func (r *OpenAIRouter) handleRouterReplaySessionsAPI(
	method string,
	rawQuery string,
) *ext_proc.ProcessingResponse {
	if method != "GET" {
		return r.createErrorResponse(405, "method not allowed")
	}
	values, err := url.ParseQuery(rawQuery)
	if err != nil {
		return r.createErrorResponse(400, "invalid query parameters")
	}
	filters, err := parseRouterReplayFilters(values)
	if err != nil {
		return r.createErrorResponse(400, err.Error())
	}
	records, err := r.queryRouterReplayDatasetRecords(filters)
	if err != nil {
		if errors.Is(err, errRouterReplayDatasetTooLarge) {
			return r.createErrorResponse(400, err.Error())
		}
		return r.createErrorResponse(500, "router replay storage query failed")
	}
	return r.createRouterReplayJSONResponse(200, routerreplay.BuildSessionDataset(records))
}

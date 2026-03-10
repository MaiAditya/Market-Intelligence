import sys
sys.path.append("/home/admin_summonlm_com/vibetrading/ai-market-intelligence")
from datetime import datetime, timezone
import json
from belief_graph.models import *
from belief_graph.llm_causal_generator import *
from utils.json_utils import NumpyJSONEncoder

belief_node = BeliefNode(
    belief_id="b1", question="q", resolution_time=datetime.now(timezone.utc),
    current_price=0.5, liquidity=0.0, event_id="e1", polymarket_slug="s"
)
en = EventNode(
    event_id="test", event_type="policy", timestamp=datetime.now(timezone.utc),
    actors=[], action="act", object="", certainty=0.5, source="LLM",
    scope="global", raw_title="t", url=None, source_doc_id=None
)
be = BeliefEdge(
    edge_id="e1", from_event_id="test", to_event_id="b1",
    mechanism_type="expectation_shift", direction="ambiguous",
    latency="uncertain", confidence=0.5, evidence=EvidenceScores(0.0,0.0,0.0,0.0), explanation=""
)

graph = BeliefGraph(belief_node=belief_node, event_nodes={"test": en}, edges=[be], depth=2)

try:
    data = graph.to_dict()
    print("to_dict OK")
    data["_storage"] = {}
    data_json = json.dumps(data, cls=NumpyJSONEncoder)
    print("json.dumps OK")
except Exception as e:
    import traceback
    traceback.print_exc()

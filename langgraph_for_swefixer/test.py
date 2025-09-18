from typing import TypedDict, get_type_hints
from langgraph.graph import StateGraph

class MyState(TypedDict, total=False):
    foo: int
    bar: str

def my_node(state: MyState) -> dict:
    # 实际逻辑无所谓，这里随便写点
    return {"foo": state.get("foo", 0) + 1}

# 建图时没提供 input_schema
builder = StateGraph(MyState)
builder.add_node("my_node", my_node)

# 看看 add_node 自动推断出的 input_schema
spec = builder.nodes["my_node"]
print(spec.input_schema)
print(get_type_hints(spec.input_schema))
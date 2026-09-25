from omnicapbench_pipeline.model_client import parse_response_to_json


def test_parse_json_object():
    parsed = parse_response_to_json('{"references": []}')
    assert parsed == {"references": []}


def test_parse_json_code_block():
    parsed = parse_response_to_json('```json\n{"events": []}\n```')
    assert parsed == {"events": []}


def test_parse_list_wrapping():
    parsed = parse_response_to_json('[{"shot_id": "shot_1"}]')
    assert parsed == {"shots": [{"shot_id": "shot_1"}]}

from telos.tokenizer import TELOS_TOKEN_MAP, TelosTokenizer
 
 
def main() -> None:
    print("loading tokenizer...")
    tt = TelosTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
 
    print("\n" + tt.describe())
    print(f"\nvocab size: {tt.vocab_size}")
 
    # each marker should encode to exactly one token at the expected ID.
    print("\n--- single-token verification ---\n")
    all_single = True
    for telos_name, _slot in TELOS_TOKEN_MAP:
        ids = tt.encode(telos_name)
        expected = tt.id_of(telos_name)
        ok = len(ids) == 1 and ids[0] == expected
        all_single = all_single and ok
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {telos_name:<14} ids={ids} expected_id={expected}")
    print(f"\nall markers single-token: {all_single}")
 
    # sample trajectory: encode, count tokens, round-trip decode.
    trajectory = (
        "<|goal|>You are a coding assistant.\n"
        "<|mission|>How many lines are in main.py?\n"
        '<|action|>{"tool":"read_file","path":"main.py"}<|end|>\n'
        '<|result|>{"ok":1,"value":"def main():\\n    print(\'hi\')\\n"}\n'
        "<|belief|>main.py has 2 lines.\n"
        '<|action|>{"tool":"answer","text":"main.py has 2 lines."}<|end|>'
    )
 
    ids = tt.encode(trajectory)
    print("\n--- sample trajectory ---\n")
    print(f"character length:     {len(trajectory)}")
    print(f"token count:     {len(ids)}")
    print(f"characters per token: {len(trajectory) / len(ids):.2f}")
    print(f"tokens: {ids}")
 
    decoded = tt.decode(ids)
    print("\n--- round-trip verification ---\n")
    matches = decoded == trajectory
    print(f"decoded matches original: {matches}")
    if not matches:
        print("\noriginal:")
        print(repr(trajectory))
        print("\ndecoded:")
        print(repr(decoded))
 
    # spot-check that the underlying tokenizer is unmodified
    print("\n--- underlying tokenizer untouched ---\n")
    raw_ids = tt.hf.encode("<|reserved_special_token_0|>", add_special_tokens=False)
    print(f"raw encode of <|reserved_special_token_0|> token: {raw_ids}")
    raw_decoded = tt.hf.decode(raw_ids, skip_special_tokens=False)
    print(f"raw decode: {raw_decoded!r}")
 
    # sanity: <|end|> stop token id is exposed correctly.
    print(f"\n<|end|> stop token id: {tt.end_id}")

    print("\n--- token IDs for telos markers ---\n")
    print(f"<|goal|> id: {tt.goal_id}")
    print(f"<|mission|> id: {tt.mission_id}")
    print(f"<|obs|> id: {tt.obs_id}")
    print(f"<|belief|> id: {tt.belief_id}")
    print(f"<|plan|> id: {tt.plan_id}")
    print(f"<|think|> id: {tt.think_id}")
    print(f"<|action|> id: {tt.action_id}")
    print(f"<|end|> id: {tt.end_id}")
    print(f"<|result|> id: {tt.result_id}")
    print(f"<|feedback|> id: {tt.feedback_id}")
    print(f"<|reward|> id: {tt.reward_id}")
 
 
if __name__ == "__main__":
    main()
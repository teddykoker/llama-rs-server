# llama-rs-server

In progress. OpenAI-style HTTP API for LLaMA, Alpaca, etc.

## Usage

Generate `ggml` models with [llama.cpp](https://github.com/ggerganov/llama.cpp).
Run with:

```bash
cargo run --release -- -m models/7B/ggml-model-q4_0.bin
```

Example usage:

```shell
$ curl --request POST \
  --url http://localhost:3000/completions \
  --header 'Content-Type: application/json' \
  --data '{
        "prompt": "Llamas are ",
        "max_tokens": 128
}'

{"text":"Llamas are 5-9 in height, with average weights of around one to two hundred pounds. They have an amazing memory and make great pets! Llama's live for anywhere between twenty five years up to thirty nine depending on the care they recieve...\nMiniature Goats (Nigerian Dwarfs) are adorable, smart animals that can grow from 20-36 inches tall. They have a great ability of adapting and surviving in harsh climates with their dense hair coating to protect them against the elements...."}
```

## References
 - [llama-rs](https://github.com/setzer22/llama-rs)
 - [llama-http](https://github.com/philpax/llama-http)
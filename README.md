# embedding-search

Local TF-IDF search over a folder of documents — no API needed and no
data leaves your machine. It builds a TF-IDF index from the text files
in a folder and ranks them by cosine similarity to your query.

## Requirements

- Python 3.8+
- No third-party packages (standard library only)

Supported file types: `.txt`, `.md`, `.rst`, `.py`, `.js`, `.ts`,
`.html`, `.csv`.

## Usage

```bash
python3 main.py docs "how do I reset my password"
python3 main.py docs -r -n 10
python3 main.py docs "deployment steps" --min-score 0.1
```

Options:

- `folder` — folder of documents to search
- `query` — search query (omit to be prompted interactively)
- `-n, --number` — number of results to show (default: 5)
- `-r, --recursive` — search subfolders too
- `--min-score` — hide results below this similarity score

## Example

```bash
$ mkdir -p /tmp/docs && printf 'Cats are small furry animals that love to sleep.\n' > /tmp/docs/cats.txt
$ printf 'Dogs are loyal animals that love walks and fetching balls.\n' > /tmp/docs/dogs.txt
$ python3 main.py /tmp/docs "furry animals that sleep"
```

Sample output:

```
1. /tmp/docs/cats.txt  (score 0.707)
2. /tmp/docs/dogs.txt  (score 0.189)
```

## License

MIT — see [LICENSE](LICENSE).

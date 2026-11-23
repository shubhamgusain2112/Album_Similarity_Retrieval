# Album Similarity Retrieval using Music Embeddings and Genre Information
LIACS, Leiden University
| Team                  | Student ID |                                      | 
| :--------             | :---:      | :------                              |
| **Binghao Guo**       |            | https://github.com/BGuoMax           |
| **Shubham Gusain**    | s4846761   | https://github.com/shubhamgusain2112 |
| **Zhou Tong**         |            | https://github.com/ztong3771         |
| **Siwen Zhang**       | s4683226   | https://github.com/ssswzh            |

### Prerequisites
We are working in Python 3.11 and I think FFmpeg will be necessary as well so install those.

Copy and paste :)

### Clone the repo

```
git clone https://github.com/shubhamgusain2112/Album_Similarity_Retrieval.git
cd .\Album_Similarity_Retrieval
```

### Set up the ```.venv```

```
python -m venv .venv
.\.venv\Scripts\activate
# or use uv
uv venv --python 3.11
uv sync
source .venv/bin/activate
```

### Dataset:
Download FMA Medium dataset [[Link]](https://github.com/mdeff/fma) and metadata, unzip them to `data/raw/` folder:
```bash
mkdir -p data/raw/
curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
python -m zipfile -e fma_metadata.zip data/raw/
curl -O https://os.unil.cloud.switch.ch/fma/fma_medium.zip
python -m zipfile -e fma_medium.zip data/raw/
```

### Directory structure
The Python package lives directly in the project root:

```text
Album_Similarity_Retrieval/
├── album_similarity/
│   ├── __init__.py
│   ├── contracts.py
│   ├── utils.py
│   ├── data.py
│   ├── audio.py
│   ├── representation.py
│   ├── retrieval.py
│   ├── evaluation.py
│   └── ui.py
├── configs/
├── data/
│   ├── raw/
│   │   ├── fma_medium/
│   │   └── fma_metadata/
│   ├── prepared/
│   ├── embeddings/
│   └── representations/
├── outputs/
├── tests/
├── pyproject.toml
└── README.md
```

The root `data/` directory stores dataset files.

### Data Preparation (Person A)

The FMA Medium data preparation pipeline has been implemented and validated.

**Results:**
- 3,397 albums and 22,528 tracks retained
- 163 genre categories
- Album-level multi-hot genre representations
- 12 unit tests passed

**Run the pipeline:**

    python -m album_similarity.data --config configs/data.yaml

**Outputs:**
- tracks.csv, albums.csv, album_metadata.json
- genres.csv, album_genre_index.csv, album_genres.npy
- quality_report.json, manifest.json

Generated files are saved in `data/prepared/`.
Raw audio and generated datasets are excluded from Git.

**Documentation:** [Data Preparation Guide](docs/data_preparation.md)

**Status:** Data preparation completed. Streamlit UI development is pending.

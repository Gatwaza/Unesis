# Unesis

Unesis is a lightweight Python SDK and toolkit for building AI applications — model orchestration, data handling, CLI utilities, and integrations used across the Unesis ecosystem. Unesis is designed to be modular so components (ASR, MT, TTS, analytics, model wrappers) can be swapped or extended easily.

---

# Project: "Bridging Medical Diagnosis Gaps in Communication and Analysis"  
Description: An AI-powered solution using Llama 3 to translate real-time medical information and analyze patient data to support timely diagnoses for displaced and underserved populations.  
Recognition: First place — prize and mentorship from Meta engineers.  
Thank you to ALU, Meta, our mentors Delali Vorgbe, Elizabeth (Liz) Ngonzi, and Mykel Kochenderfer, and to everyone who participated.  

Team: Jean Robert Gatwaza, Christine Akoto-Nimoh, Muwanguzi Arnold, Joyce Moses Brown

This README and the Unesis toolkit now include artifacts, integrations, and documentation produced during the hackathon. See the "Hackathon Project" section below for links and pointers.

---

## Features

- Model orchestration helpers for local and hosted models
- Data preprocessing utilities (audio, text)
- Lightweight CLI for common tasks (preprocess, run, test)
- Pluggable backends: ASR, MT, TTS, classification, and LLM analysis
- Docker-ready for reproducible deployments
- Example pipelines (audio translation, clinical text analysis)

---

## Hackathon Project: Bridging Medical Diagnosis Gaps

The hackathon solution demonstrates how Unesis can be used to build real-world AI workflows:

- Real-time ASR -> translation -> clinical text normalization
- LLM (Llama 3) analysis for triage and diagnostic suggestions
- Output structured clinical summaries and diagnostic indicators for clinicians and humanitarian responders
- Designed for low-bandwidth and privacy-conscious deployments

Related repositories:
- Audio translation pipeline: https://github.com/Unesis-AI/Audio-Translation
- Project website / docs: https://github.com/Unesis-AI/Unesis-AI.github.io

If you want the hackathon demo code and notebooks integrated into this repo, open an issue or a PR and we’ll coordinate adding them into examples/.

---

## Quickstart

Clone and install:

```bash
git clone https://github.com/Gatwaza/Unesis.git
cd Unesis
python -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows
pip install -r requirements.txt
pip install -e .
```

Run the CLI help:

```bash
python -m unesis.cli --help
```

Example: run a demonstration pipeline (audio -> text -> translate -> LLM analyze)

```bash
python -m unesis.cli run-pipeline \
  --pipeline examples/pipelines/clinical_audio_translate.yaml \
  --input data/example_patient.wav \
  --device cpu
```

(See examples/pipelines/ for YAML pipeline configs. Swap device to `cuda` if GPU is available.)

---

## Configuration and Secrets

- Default config in `config.yaml`
- Keep API keys and secrets out of the repo. Use environment variables or a `.env` file and add it to `.gitignore`.
- Example environment variables:
  - UNESIS_LLM_API_KEY
  - UNESIS_TTS_KEY
  - UNESIS_ASR_KEY

---

## Development

Run tests:

```bash
pytest tests/
```

Code style:

```bash
black .
flake8 .
```

Add new backends:
- Implement the backend interface in `unesis/backends/`
- Register new backend in `unesis/config.py` or your pipeline config YAML

Docker:

```bash
docker build -t unesis:latest .
docker run --rm -v $(pwd):/workspace unesis:latest python -m unesis.cli run-pipeline --pipeline examples/...
```

---

## Examples and Notebooks

The repo includes example pipelines and Jupyter notebooks (see `examples/` and `notebooks/`) demonstrating:
- Audio transcription + translation
- LLM-based clinical note summarization (Llama 3 integration)
- Evaluation scripts and sample test data

---

## Contributing

We welcome contributions. Typical workflow:
1. Open an issue describing the change or feature.
2. Fork the repo and create a feature branch.
3. Add tests and documentation for changes.
4. Open a pull request referencing the issue.

Please follow the code style, add tests for new functionality, and include a short changelog entry.

---

## License

This repository is released under the MIT License. See LICENSE for details.

---

## Contact

Maintainers: Jean Robert Gatwaza and Chol Daniel Deng Dau 
Repo: https://github.com/Gatwaza/Unesis  
For hackathon or partnership inquiries: contact@unesis.ai (or open an issue)

---

## Acknowledgements

Thanks to ALU Rwanda and Meta for organizing the symposium and hackathon and for their mentorship and support. Special thanks to our mentors Delali Vorgbe, Elizabeth (Liz) Ngonzi, and Mykel Kochenderfer.

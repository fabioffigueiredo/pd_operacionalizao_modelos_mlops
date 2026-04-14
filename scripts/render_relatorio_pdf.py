#!/usr/bin/env python3
"""
Gera um PDF do relatório técnico a partir do Markdown.

Estratégia:
1. Converte `reports/relatorio_tecnico.md` para HTML com Python-Markdown
2. Aplica CSS de impressão em A4
3. Usa Google Chrome headless para imprimir em PDF

Saídas:
- .render_tmp/reports/relatorio_tecnico_print.html
- reports/relatorio_tecnico.pdf
"""

from __future__ import annotations

import base64
import html
import shutil
import subprocess
import tempfile
from pathlib import Path

try:
    import markdown
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Dependência ausente: instale `Markdown` com `python3 -m pip install --user markdown`."
    ) from exc

try:
    from pypdf import PdfReader, PdfWriter
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Dependência ausente: instale `pypdf` com `python3 -m pip install --user pypdf`."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = PROJECT_ROOT / "reports" / "relatorio_tecnico.md"
LOCAL_ARTIFACTS_DIR = PROJECT_ROOT / ".render_tmp" / "reports"
REPORT_HTML = LOCAL_ARTIFACTS_DIR / "relatorio_tecnico_print.html"
REPORT_PDF = PROJECT_ROOT / "reports" / "relatorio_tecnico.pdf"
LOGO_PATH = PROJECT_ROOT / "pd-ml-scikit-learning-main" / "images" / "logo_infnet.png"
REPO_URL = "https://github.com/fabioffigueiredo/pd_operacionalizao_modelos_mlops"
VIDEO_DRIVE_URL = "https://drive.google.com/file/d/11Yn6D01kEwuc6N-__t40ZlxxyOoEa-uM/view?usp=sharing"


def get_chrome_binary() -> str:
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise FileNotFoundError("Google Chrome/Chromium não encontrado para exportar o PDF.")


def markdown_to_html(markdown_text: str) -> str:
    return markdown.markdown(
        markdown_text,
        extensions=["extra", "tables", "fenced_code", "sane_lists", "toc"],
        output_format="html5",
    )


def encode_logo() -> str:
    data = LOGO_PATH.read_bytes()
    return base64.b64encode(data).decode("ascii")


def split_body(markdown_text: str) -> str:
    marker = "\n---\n"
    if marker in markdown_text:
        return markdown_text.split(marker, 1)[1].lstrip()
    return markdown_text


def build_cover_html(logo_b64: str) -> str:
    return f"""
    <section class="page cover">
      <img class="cover-logo" src="data:image/png;base64,{logo_b64}" alt="Instituto Infnet logo">
      <div class="cover-program-group">
        <p class="cover-program">Pós-Graduação em Machine Learning,</p>
        <p class="cover-program">Deep Learning e Inteligência Artificial</p>
      </div>
      <p class="cover-discipline">Operacionalização de Modelos com MLOps [26E1_2]</p>
      <p class="cover-professor">Prof. Ícaro Augusto Maccari Zelioli</p>
      <div class="cover-bottom">
        <p class="cover-title">Avaliação de Risco de Crédito</p>
        <p class="cover-date">12 de abril de 2026</p>
        <p class="cover-subtitle">Projeto: Operacionalização de Modelos com MLflow e Streamlit</p>
      </div>
    </section>
    """


def build_summary_html() -> str:
    return f"""
    <section class="page summary">
      <div class="summary-block">
        <p class="summary-label">Aluno:</p>
        <ul>
          <li>Fabio Ferreira Figueiredo</li>
        </ul>
      </div>

      <div class="summary-block">
        <p class="summary-label">Link GitHub:</p>
        <p><a href="{html.escape(REPO_URL)}">{html.escape(REPO_URL)}</a></p>
      </div>

      <div class="summary-block">
        <p class="summary-label">Vídeo de demonstração:</p>
        <p>Arquivo local no repositório: <code>mlflow+streamlit_mlops.mp4</code></p>
        <p>Backup Google Drive: <a href="{html.escape(VIDEO_DRIVE_URL)}">{html.escape(VIDEO_DRIVE_URL)}</a></p>
      </div>

      <div class="summary-block">
        <h2>Descrição do Problema, MLOps e Pipeline</h2>
        <h3>Contexto do Problema:</h3>
        <p>
          O dataset <strong>Give Me Some Credit</strong> contém informações de aproximadamente 150.000
          tomadores de crédito. O objetivo é prever inadimplência severa nos próximos 2 anos,
          usando a variável alvo binária <code>SeriousDlqin2yrs</code>.
        </p>
        <p>
          Nesta segunda etapa, o foco deixa de ser apenas comparação de modelos e passa a ser
          <strong>operacionalização</strong>: organização de código, rastreabilidade com MLflow,
          seleção controlada de modelo campeão e inferência reproduzível via Streamlit.
        </p>
        <p>
          O contexto regulatório continua central. Em crédito, a solução precisa equilibrar
          desempenho, interpretabilidade e governança, considerando exigências associadas à
          LGPD e à transparência esperada em decisões automatizadas.
        </p>
      </div>

      <div class="summary-block">
        <h2>Guia de Navegação — Rubrica de Avaliação</h2>
        <div class="summary-note">
          <strong>Nota ao avaliador:</strong> a estrutura do relatório foi reorganizada para seguir
          diretamente a ordem da rubrica do professor. Cada competência aparece como uma seção principal,
          facilitando a validação do que foi demonstrado em cada critério.
        </div>

        <table class="nav-table">
          <thead>
            <tr>
              <th>Fase da Rubrica</th>
              <th>Competência Avaliada</th>
              <th>Seção do Relatório</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Competência 1</td>
              <td>Estruturação do projeto, contexto, planejamento e transição do notebook para entrega</td>
              <td>Seção 1</td>
            </tr>
            <tr>
              <td>Competência 2</td>
              <td>Pipelines de dados, qualidade, features e análise de redução de dimensionalidade</td>
              <td>Seção 2</td>
            </tr>
            <tr>
              <td>Competência 3</td>
              <td>Experimentos reprodutíveis, comparação de abordagens e rastreamento com MLflow</td>
              <td>Seção 3</td>
            </tr>
            <tr>
              <td>Competência 4</td>
              <td>Operacionalização, inferência, métricas de negócio, monitoramento, drift e re-treino</td>
              <td>Seção 4</td>
            </tr>
            <tr>
              <td>Fechamento</td>
              <td>Conclusões e próximos passos</td>
              <td>Seção 5</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
    """


def build_html(markdown_text: str) -> str:
    body = markdown_to_html(split_body(markdown_text))
    logo_b64 = encode_logo()
    title = "Relatório Técnico — PD2 MLOps"
    css = """
    @page {
      size: A4;
      margin: 16mm 14mm 18mm 14mm;
    }
    :root {
      color-scheme: light;
      --ink: #1f2937;
      --muted: #6b7280;
      --line: #d1d5db;
      --bg-soft: #f8fafc;
      --accent: #0f4c81;
    }
    * { box-sizing: border-box; }
    body {
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      line-height: 1.45;
      font-size: 11px;
      margin: 0;
      padding: 0;
    }
    main {
      max-width: 100%;
      margin: 0 auto;
    }
    .page {
      page-break-after: always;
    }
    .page:last-child {
      page-break-after: auto;
    }
    .cover {
      position: relative;
      height: 250mm;
      text-align: center;
      overflow: hidden;
    }
    .cover-logo {
      position: absolute;
      top: 10mm;
      left: 50%;
      transform: translateX(-50%);
      width: 40mm;
      height: auto;
    }
    .cover-program-group {
      position: absolute;
      top: 52mm;
      left: 50%;
      transform: translateX(-50%);
      width: 130mm;
    }
    .cover-program,
    .cover-discipline,
    .cover-professor,
    .cover-title,
    .cover-date,
    .cover-subtitle {
      margin: 0;
      font-weight: 700;
      color: #111827;
    }
    .cover-program { font-size: 16px; }
    .cover-discipline {
      position: absolute;
      top: 128mm;
      left: 50%;
      transform: translateX(-50%);
      width: 150mm;
      font-size: 16px;
      color: #19324a;
    }
    .cover-professor {
      position: absolute;
      top: 150mm;
      left: 50%;
      transform: translateX(-50%);
      width: 120mm;
      font-size: 16px;
    }
    .cover-bottom {
      position: absolute;
      bottom: 12mm;
      left: 50%;
      transform: translateX(-50%);
      width: 170mm;
    }
    .cover-title { font-size: 19px; }
    .cover-date { font-size: 18px; margin-top: 16mm; }
    .cover-subtitle { font-size: 18px; margin-top: 2mm; }
    .summary {
      position: relative;
      height: 240mm;
      padding-top: 6mm;
      overflow: hidden;
    }
    .summary-block {
      margin-bottom: 8mm;
    }
    .summary-label {
      font-weight: 700;
      margin: 0 0 4mm;
    }
    .summary ul {
      margin: 0;
      padding-left: 6mm;
    }
    .summary h2 {
      margin: 0 0 4mm;
      font-size: 18px;
      color: #111827;
      border: none;
      padding: 0;
    }
    .summary h3 {
      margin: 0 0 2mm;
      font-size: 15px;
      color: #111827;
    }
    .summary p,
    .summary li {
      font-size: 13px;
      line-height: 1.4;
    }
    .summary-note {
      border-left: 4px solid #d1d5db;
      padding-left: 4mm;
      margin: 3mm 0 5mm;
      font-size: 12.5px;
    }
    .nav-table th,
    .nav-table td {
      font-size: 11px;
      padding: 7px 8px;
    }
    .report-body h1 {
      margin-top: 0;
      font-size: 23px;
    }
    h1, h2, h3, h4 {
      color: #0b2942;
      margin-top: 1.2em;
      margin-bottom: 0.35em;
      line-height: 1.2;
      page-break-after: avoid;
    }
    h1 {
      font-size: 22px;
      border-bottom: 2px solid var(--accent);
      padding-bottom: 8px;
      margin-top: 0;
    }
    h2 { font-size: 16px; }
    h3 { font-size: 13px; }
    h4 { font-size: 12px; }
    p, li {
      orphans: 3;
      widows: 3;
    }
    p { margin: 0.45em 0; }
    ul, ol {
      margin: 0.4em 0 0.7em 1.25em;
      padding: 0;
    }
    li { margin: 0.18em 0; }
    hr {
      border: none;
      border-top: 1px solid var(--line);
      margin: 1.2em 0;
    }
    code {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.94em;
      background: #eef2f7;
      padding: 0.1em 0.28em;
      border-radius: 4px;
    }
    pre {
      background: #0f172a;
      color: #e5e7eb;
      padding: 12px;
      border-radius: 8px;
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-word;
      page-break-inside: avoid;
    }
    pre code {
      background: transparent;
      padding: 0;
      color: inherit;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin: 0.8em 0 1.1em;
      page-break-inside: auto;
    }
    thead {
      display: table-header-group;
    }
    tr {
      page-break-inside: avoid;
    }
    th, td {
      border: 1px solid var(--line);
      padding: 6px 7px;
      vertical-align: top;
      text-align: left;
      font-size: 10px;
    }
    th {
      background: #eaf1f8;
      color: #102a43;
      font-weight: 600;
    }
    blockquote {
      border-left: 4px solid var(--accent);
      margin: 0.8em 0;
      padding: 0.2em 0 0.2em 0.9em;
      color: #334155;
      background: var(--bg-soft);
    }
    """
    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>{css}</style>
</head>
<body>
  <main>
    {build_cover_html(logo_b64)}
    {build_summary_html()}
    <section class="report-body">
      {body}
    </section>
  </main>
</body>
</html>
"""


def render_pdf(html_path: Path, pdf_path: Path) -> None:
    chrome = get_chrome_binary()
    with tempfile.TemporaryDirectory(prefix="pd2-pdf-") as tmp_dir:
        subprocess.run(
            [
                chrome,
                "--headless=new",
                "--disable-gpu",
                "--allow-file-access-from-files",
                "--no-pdf-header-footer",
                f"--user-data-dir={tmp_dir}",
                f"--print-to-pdf={pdf_path}",
                html_path.as_uri(),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def remove_blank_pages(pdf_path: Path) -> None:
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()

    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        if text:
            writer.add_page(page)

    with pdf_path.open("wb") as fp:
        writer.write(fp)


def main() -> None:
    if not REPORT_MD.exists():
        raise FileNotFoundError(f"Relatório não encontrado: {REPORT_MD}")

    markdown_text = REPORT_MD.read_text(encoding="utf-8")
    LOCAL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_HTML.write_text(build_html(markdown_text), encoding="utf-8")
    render_pdf(REPORT_HTML, REPORT_PDF)
    remove_blank_pages(REPORT_PDF)

    print(REPORT_HTML)
    print(REPORT_PDF)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        raise SystemExit(f"Falha ao exportar PDF: {exc}") from exc

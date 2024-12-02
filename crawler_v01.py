import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import logging
from readability import Document
import markdownify
import shutil
import re
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import time
import sys
import asyncio
from enum import Enum
from pathlib import Path
import json
from pdf2image import convert_from_path 
import pytesseract
from PIL import Image
import cv2
import numpy as np
import fitz  # PyMuPDF
import pandas as pd
import tabula
from table2ascii import table2ascii
import camelot
from datetime import datetime
import traceback
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Union

# ---------------------------- WebCrawler Class ---------------------------- #

class WebCrawler:
    def __init__(self, start_url, max_depth, delay=1):
        """
        Initialise le crawler avec l'URL de départ et la profondeur maximale.
        
        :param start_url: URL de départ pour le crawling.
        :param max_depth: Profondeur maximale de crawling.
        :param delay: Délai entre les requêtes pour éviter de surcharger le serveur.
        """
        self.start_url = self.normalize_url(start_url)
        if not self.start_url:
            logging.error("URL de départ invalide.")
            sys.exit(1)
        self.max_depth = max_depth
        self.delay = delay
        self.visited_urls = set()
        self.to_visit = [(self.start_url, 0)]
        self.all_urls = set()
        self.pdf_links = set()
        self.image_links = set()
        self.doc_links = set()
        self.setup_logging()
        self.setup_directories()
        self.session = self.create_session()
        self.domain = urlparse(self.start_url).netloc
        self.language = self.detect_language(self.start_url)
        logging.info(f"Domaine cible: {self.domain}, Langue cible: {self.language}")

    def setup_logging(self):
        """
        Configure le système de logging pour enregistrer les activités du crawler.
        """
        logging.basicConfig(
            filename='crawler_report.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.info('Crawler initialisé.')

    def setup_directories(self):
        """
        Crée les répertoires nécessaires pour stocker les fichiers téléchargés et le contenu extrait.
        """
        directories = ['PDF', 'Image', 'Doc', 'Content']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        logging.info('Répertoires créés ou déjà existants.')

    def create_session(self):
        """
        Crée une session requests avec une stratégie de retry pour gérer les requêtes échouées.
        """
        session = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def detect_language(self, url):
        """
        Détecte la langue de l'URL de départ en fonction du chemin ou des sous-domaines.
        
        :param url: URL à analyser.
        :return: Code de langue détecté ou 'unknown'.
        """
        parsed = urlparse(url)
        # Exemple simple: recherche d'un code de langue dans le chemin (e.g., /fr/, /en/)
        match = re.search(r'/([a-z]{2})/', parsed.path)
        if match:
            return match.group(1)
        # Alternative: vérification des sous-domaines (e.g., fr.example.com)
        subdomains = parsed.hostname.split('.')
        if len(subdomains) > 2:
            lang = subdomains[0]
            if len(lang) == 2:
                return lang
        return 'unknown'

    def normalize_url(self, url):
        """
        Normalise l'URL en supprimant les fragments et les paramètres, et en assurant une structure cohérente.

        :param url: URL à normaliser.
        :return: URL normalisée ou None si l'URL est invalide.
        """
        parsed = urlparse(url)
        if parsed.scheme not in ['http', 'https']:
            return None
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        normalized = re.sub(r'/+', '/', normalized)  # Remplace les slashes multiples par un seul
        return normalized.rstrip('/')

    def is_within_domain_and_language(self, url):
        """
        Vérifie si l'URL appartient au même domaine et à la même langue que l'URL de départ.

        :param url: URL à vérifier.
        :return: True si l'URL est dans le même domaine et langue, False sinon.
        """
        parsed = urlparse(url)
        if parsed.netloc != self.domain:
            return False
        url_language = self.detect_language(url)
        if self.language != 'unknown' and url_language != self.language:
            return False
        return True

    def crawl(self):
        """
        Démarre le processus de crawling en suivant une approche en pipeline.
        """
        logging.info('Démarrage du crawl.')
        start_time = time.time()
        while self.to_visit:
            current_url, depth = self.to_visit.pop(0)
            if current_url in self.visited_urls:
                continue
            if depth > self.max_depth:
                continue
            self.visited_urls.add(current_url)
            logging.info(f'Visite URL: {current_url} à la profondeur {depth}')
            try:
                response = self.session.get(current_url, timeout=10)
                time.sleep(self.delay)  # Respect du délai entre les requêtes
                if response.status_code != 200:
                    logging.warning(f'URL {current_url} retournée avec le statut {response.status_code}')
                    continue
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' not in content_type:
                    logging.info(f'URL {current_url} ignorée car le contenu n\'est pas HTML.')
                    continue
                soup = BeautifulSoup(response.content, 'html.parser')
                # Extraction des liens
                self.extract_links(current_url, soup, depth)
                # Extraction du contenu principal
                self.extract_content(current_url, response.content)
                # Téléchargement des ressources
                self.find_and_download_resources(current_url, soup)
            except requests.exceptions.RequestException as e:
                logging.error(f'Erreur lors de la requête de {current_url}: {e}')
            except Exception as e:
                logging.error(f'Erreur inattendue avec {current_url}: {e}')
        elapsed_time = time.time() - start_time
        logging.info('Crawl terminé.')
        self.generate_report(elapsed_time)

    def extract_links(self, base_url, soup, current_depth):
        """
        Extrait et normalise les liens trouvés sur la page actuelle et les ajoute à la liste de visite.

        :param base_url: URL de base pour résoudre les URLs relatives.
        :param soup: Objet BeautifulSoup de la page actuelle.
        :param current_depth: Profondeur actuelle dans le crawling.
        """
        for link_tag in soup.find_all('a', href=True):
            href = link_tag['href']
            href = urljoin(base_url, href)
            href = self.normalize_url(href)
            if href and self.is_within_domain_and_language(href):
                if href not in self.all_urls:
                    self.all_urls.add(href)
                    self.to_visit.append((href, current_depth + 1))
        logging.info(f'Liens extraits de {base_url}: {len(self.all_urls)} URLs trouvées jusqu\'à présent.')

    def extract_content(self, url, html_content):
        """
        Extrait le contenu principal de la page et le sauvegarde au format Markdown.

        :param url: URL de la page.
        :param html_content: Contenu HTML de la page.
        """
        try:
            doc = Document(html_content)
            content_html = doc.summary(html_partial=True)
            content_markdown = markdownify.markdownify(content_html, heading_style="ATX")
            # Vérification et formatage des liens dans le Markdown
            content_markdown = self.validate_and_format_links(content_markdown, url)
            filename = self.url_to_filename(url) + '.txt'
            file_path = os.path.join('Content', filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content_markdown)
            logging.info(f'Contenu extrait et sauvegardé pour {url}')
        except Exception as e:
            logging.error(f'Échec de l\'extraction du contenu de {url}: {e}')

    def validate_and_format_links(self, markdown_content, base_url):
        """
        Valide et formate les liens dans le contenu Markdown.

        :param markdown_content: Contenu en Markdown.
        :param base_url: URL de base pour résoudre les liens relatifs.
        :return: Contenu Markdown avec liens validés et formatés.
        """
        # Utilisation de regex pour trouver les liens Markdown
        pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
        matches = pattern.findall(markdown_content)
        for text, link in matches:
            # Résolution des liens relatifs
            resolved_link = urljoin(base_url, link)
            resolved_link = self.normalize_url(resolved_link)
            if resolved_link and self.is_within_domain_and_language(resolved_link):
                # Conversion en lien relatif si possible
                if resolved_link.startswith(self.start_url):
                    relative_link = os.path.relpath(resolved_link, self.start_url)
                    markdown_content = markdown_content.replace(f']({link})', f']({relative_link})')
                else:
                    # Lien absolu mais validé
                    markdown_content = markdown_content.replace(f']({link})', f']({resolved_link})')
            else:
                # Lien invalide ou hors domaine/langue, suppression ou marquage
                markdown_content = markdown_content.replace(f']({link})', f'](invalid-link)')
                logging.warning(f'Lien invalide ou hors domaine/langue dans Markdown: {link}')
        return markdown_content

    def find_and_download_resources(self, base_url, soup):
        """
        Recherche et télécharge les ressources (PDF, images, documents) trouvées sur la page.

        :param base_url: URL de base pour résoudre les URLs relatives.
        :param soup: Objet BeautifulSoup de la page actuelle.
        """
        # Téléchargement des PDFs et documents
        for link_tag in soup.find_all('a', href=True):
            href = link_tag['href']
            href = urljoin(base_url, href)
            href = href.split('#')[0].split('?')[0]
            lower_href = href.lower()
            if lower_href.endswith('.pdf'):
                if href not in self.pdf_links and self.is_within_domain_and_language(href):
                    self.pdf_links.add(href)
                    self.download_file(href, 'PDF')
            elif lower_href.endswith(('.doc', '.docx')):
                if href not in self.doc_links and self.is_within_domain_and_language(href):
                    self.doc_links.add(href)
                    self.download_file(href, 'Doc')

        # Téléchargement des images
        for img_tag in soup.find_all('img', src=True):
            src = img_tag['src']
            src = urljoin(base_url, src)
            src = src.split('#')[0].split('?')[0]
            lower_src = src.lower()
            if lower_src.endswith(('.png', '.jpg', '.jpeg')):
                if src not in self.image_links and self.is_within_domain_and_language(src):
                    self.image_links.add(src)
                    self.download_file(src, 'Image')

    def download_file(self, url, folder):
        """
        Télécharge un fichier depuis une URL et le sauvegarde dans le dossier spécifié.

        :param url: URL du fichier à télécharger.
        :param folder: Nom du dossier où sauvegarder le fichier.
        """
        try:
            response = self.session.get(url, stream=True, timeout=10)
            time.sleep(self.delay)  # Respect du délai entre les requêtes
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '').lower()
                if (folder == 'PDF' and 'application/pdf' in content_type) or \
                   (folder == 'Image' and 'image' in content_type) or \
                   (folder == 'Doc' and ('application/msword' in content_type or 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' in content_type)):
                    local_filename = self.url_to_filename(url)
                    file_extension = os.path.splitext(urlparse(url).path)[1]
                    if not local_filename.endswith(file_extension):
                        local_filename += file_extension
                    file_path = os.path.join(folder, local_filename)
                    with open(file_path, 'wb') as f:
                        shutil.copyfileobj(response.raw, f)
                    logging.info(f'Téléchargé {url} dans le dossier {folder}')
                else:
                    logging.warning(f'Type de contenu non supporté pour {url}: {content_type}')
            else:
                logging.error(f'Échec du téléchargement {url}: Code HTTP {response.status_code}')
        except requests.exceptions.RequestException as e:
            logging.error(f'Erreur lors du téléchargement de {url}: {e}')
        except Exception as e:
            logging.error(f'Erreur inattendue lors du téléchargement de {url}: {e}')

    def url_to_filename(self, url):
        """
        Convertit une URL en un nom de fichier valide.

        :param url: URL à convertir.
        :return: Nom de fichier sécurisé.
        """
        parsed_url = urlparse(url)
        path = parsed_url.path
        filename = path.strip('/').replace('/', '_')
        if not filename:
            filename = 'index'
        # Supprime les caractères non autorisés
        filename = re.sub(r'[^\w\-_\. ]', '_', filename)
        return filename

    def generate_report(self, elapsed_time):
        """
        Génère un rapport résumé du processus de crawling.

        :param elapsed_time: Temps total écoulé en secondes.
        """
        logging.info('Génération du rapport final.')
        report_lines = [
            'Rapport de Crawl',
            '================',
            f'URL de départ: {self.start_url}',
            f'Profondeur maximale: {self.max_depth}',
            f'Total des URLs visitées : {len(self.visited_urls)}',
            f'Total des PDFs téléchargés : {len(self.pdf_links)}',
            f'Total des images téléchargées : {len(self.image_links)}',
            f'Total des documents téléchargés : {len(self.doc_links)}',
            f'Temps total écoulé: {elapsed_time:.2f} secondes'
        ]
        report = '\n'.join(report_lines)
        with open('crawl_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        logging.info('Rapport généré avec succès.')

# ------------------------ PDF Processing Classes ------------------------- #

# Configuration avancée pour le traitement des PDF
@dataclass
class PDFProcessingConfig:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = "gpt-4"
    OCR_LANGUAGE: str = "eng"
    MAX_TOKENS: int = 2048
    TEMPERATURE: float = 0.3
    OUTPUT_FORMAT: str = "markdown"
    TABLE_EXTRACTION_METHODS: List[str] = ("camelot", "tabula", "ocr")
    MIN_TABLE_CONFIDENCE: float = 0.7
    ENABLE_TABLE_RESTRUCTURING: bool = True
    TABLE_FORMAT_STYLE: str = "markdown"  # or "ascii"
    IMAGE_DPI: int = 300
    OCR_CONFIG: Dict[str, str] = None

    def __post_init__(self):
        if self.OCR_CONFIG is None:
            self.OCR_CONFIG = {
                "lang": self.OCR_LANGUAGE,
                "config": "--oem 3 --psm 6"
            }

class OpenAIProcessor:
    def __init__(self, config: PDFProcessingConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {config.OPENAI_API_KEY}",
            "Content-Type": "application/json"
        })

    def process_text(self, text: str, instruction: Optional[str] = None) -> Optional[str]:
        try:
            if not instruction:
                instruction = "Format and correct the following text into well-structured markdown."

            data = {
                "model": self.config.OPENAI_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": instruction
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                "temperature": self.config.TEMPERATURE,
                "max_tokens": self.config.MAX_TOKENS
            }

            response = self.session.post(
                "https://api.openai.com/v1/chat/completions",
                json=data
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']

        except Exception as e:
            logging.error(f"Error in OpenAI processing: {e}")
            return None

    def restructure_table(self, table_data: 'TableData') -> Optional[str]:
        try:
            # Convert table data to string representation
            table_str = "Headers: " + ", ".join(table_data.headers) + "\n"
            for row in table_data.rows:
                table_str += "Row: " + ", ".join(row) + "\n"

            instruction = """
            Restructure this table data into a well-formatted markdown structure.
            For each row, create a section with:
            1. A header for the main item
            2. Bullet points for other columns
            3. Ensure proper formatting and capitalization
            Example format:
            # [First Column Value]
            - [Second Column Header]: [Second Column Value]
            - [Third Column Header]: [Third Column Value]
            """

            return self.process_text(table_str, instruction)

        except Exception as e:
            logging.error(f"Error in table restructuring: {e}")
            return None

@dataclass
class TableData:
    headers: List[str]
    rows: List[List[str]]
    page_number: int
    position: Dict[str, float]  # x1, y1, x2, y2
    confidence_score: float

@dataclass
class PDFMetadata:
    file_path: Path
    pdf_type: 'PDFType'
    page_count: int
    has_images: bool
    has_text: bool
    has_tables: bool
    file_size: int
    creation_date: str
    content_types: List['ContentType']
    table_count: int
    image_count: int

class PDFType(Enum):
    SEARCHABLE = "searchable"
    SCANNED = "scanned"
    MIXED = "mixed"
    IMAGE_ONLY = "image_only"
    TABLE_HEAVY = "table_heavy"

class ContentType(Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    MIXED = "mixed"

class TableExtractor:
    def __init__(self, config: PDFProcessingConfig):
        self.config = config

    def extract_tables_camelot(self, pdf_path: Path) -> List[TableData]:
        try:
            tables = camelot.read_pdf(str(pdf_path), pages='all')
            return [self._convert_to_table_data(table, idx + 1) for idx, table in enumerate(tables)]
        except Exception as e:
            logging.error(f"Error extracting tables with Camelot: {e}")
            return []

    def extract_tables_tabula(self, pdf_path: Path) -> List[TableData]:
        try:
            tables = tabula.read_pdf(str(pdf_path), pages='all', multiple_tables=True)
            return [self._convert_to_table_data(table, idx + 1) for idx, table in enumerate(tables)]
        except Exception as e:
            logging.error(f"Error extracting tables with Tabula: {e}")
            return []

    def extract_tables_ocr(self, image: Image.Image, page_number: int) -> List[TableData]:
        try:
            # Prétraitement de l'image
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # Détection des lignes horizontales et verticales
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
            horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

            # Combinaison des lignes
            table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            
            # Trouver les contours
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            tables = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 100 and h > 100:  # Filtrer les petits contours
                    table_region = image.crop((x, y, x+w, y+h))
                    text = pytesseract.image_to_string(table_region, config=self.config.OCR_CONFIG['config'])
                    
                    # Analyse basique du texte pour extraire les données du tableau
                    rows = text.split('\n')
                    table_data = []
                    for row in rows:
                        if row.strip():
                            cells = row.split()
                            if cells:
                                table_data.append(cells)
                    
                    if len(table_data) > 1:  # Au moins une en-tête et une ligne
                        tables.append(TableData(
                            headers=table_data[0],
                            rows=table_data[1:],
                            page_number=page_number,
                            position={"x1": x, "y1": y, "x2": x+w, "y2": y+h},
                            confidence_score=0.8  # Score arbitraire, à affiner
                        ))
            
            return tables
        except Exception as e:
            logging.error(f"Error in OCR table extraction: {e}")
            return []

    def _convert_to_table_data(self, table, page_number: int) -> TableData:
        try:
            if isinstance(table, pd.DataFrame):
                headers = table.columns.tolist()
                rows = table.values.tolist()
            else:
                # Handle camelot table format
                headers = table.df.iloc[0].tolist()
                rows = table.df.iloc[1:].values.tolist()

            # Nettoyage des données
            headers = [str(h).strip() for h in headers]
            rows = [[str(cell).strip() for cell in row] for row in rows]

            # Calcul du score de confiance basé sur la qualité des données
            confidence_score = self._calculate_confidence_score(headers, rows)

            return TableData(
                headers=headers,
                rows=rows,
                page_number=page_number,
                position={"x1": 0, "y1": 0, "x2": 0, "y2": 0},
                confidence_score=confidence_score
            )
        except Exception as e:
            logging.error(f"Error converting table to TableData: {e}")
            return TableData(headers=[], rows=[], page_number=page_number, position={}, confidence_score=0.0)

    def _calculate_confidence_score(self, headers: List[str], rows: List[List[str]]) -> float:
        try:
            # Vérification de la cohérence des données
            if not headers or not rows:
                return 0.0

            # Vérification de la longueur des lignes
            expected_length = len(headers)
            length_consistency = sum(1 for row in rows if len(row) == expected_length) / len(rows)

            # Vérification de la qualité des données
            empty_cells = sum(1 for row in rows for cell in row if not cell.strip())
            total_cells = len(rows) * len(headers)
            data_quality = 1 - (empty_cells / total_cells if total_cells > 0 else 0)

            # Score final
            return min(1.0, (length_consistency * 0.6 + data_quality * 0.4))

        except Exception as e:
            logging.error(f"Error calculating confidence score: {e}")
            return 0.0

class PDFProcessor(ABC):
    def __init__(self, config: PDFProcessingConfig):
        self.config = config
        self.openai_processor = OpenAIProcessor(config)
        self.table_extractor = TableExtractor(config)

    @abstractmethod
    async def process(self, pdf_path: Path) -> Dict[str, Union[str, List[TableData]]]:
        pass

    async def process_tables(self, pdf_path: Path, images: Optional[List[Image.Image]] = None) -> List[TableData]:
        tables = []
        for method in self.config.TABLE_EXTRACTION_METHODS:
            try:
                if method == "camelot":
                    extracted_tables = self.table_extractor.extract_tables_camelot(pdf_path)
                    tables.extend(extracted_tables)
                elif method == "tabula":
                    extracted_tables = self.table_extractor.extract_tables_tabula(pdf_path)
                    tables.extend(extracted_tables)
                elif method == "ocr" and images:
                    for page_number, image in enumerate(images, start=1):
                        extracted_tables = self.table_extractor.extract_tables_ocr(image, page_number)
                        tables.extend(extracted_tables)
            except Exception as e:
                logging.warning(f"Failed to extract tables using {method}: {e}")

        # Filtrer les tables selon le score de confiance
        filtered_tables = [
            table for table in tables 
            if table.confidence_score >= self.config.MIN_TABLE_CONFIDENCE
        ]

        # Déduplication des tables
        unique_tables = self._deduplicate_tables(filtered_tables)

        return unique_tables

    def _deduplicate_tables(self, tables: List[TableData]) -> List[TableData]:
        """Supprime les tables en double basé sur leur contenu."""
        unique_tables = []
        seen_contents = set()

        for table in tables:
            # Créer une représentation hashable du contenu de la table
            content_hash = str(table.headers) + str(table.rows)
            
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_tables.append(table)

        return unique_tables

class SearchablePDFProcessor(PDFProcessor):
    async def process(self, pdf_path: Path) -> Dict[str, Union[str, List[TableData]]]:
        try:
            doc = fitz.open(pdf_path)
            per_page_text = {}
            tables = await self.process_tables(pdf_path)

            # Extraction du texte page par page
            for page in doc:
                page_number = page.number + 1  # Les pages commencent à 0
                text_content = page.get_text()
                processed_text = self.openai_processor.process_text(text_content)
                per_page_text[page_number] = processed_text

            # Restructuration des tables si activée
            if self.config.ENABLE_TABLE_RESTRUCTURING and tables:
                for table in tables:
                    restructured = self.openai_processor.restructure_table(table)
                    if restructured:
                        page_text = per_page_text.get(table.page_number, "")
                        per_page_text[table.page_number] += "\n\n" + restructured

            return {
                "text": per_page_text,  # Dictionnaire avec clé page_number et valeur texte
                "tables": tables
            }
        except Exception as e:
            logging.error(f"Error processing searchable PDF: {e}")
            return {"text": "", "tables": []}

class ScannedPDFProcessor(PDFProcessor):
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        # Conversion en array numpy
        img_array = np.array(image)
        
        # Conversion en niveaux de gris
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Débruitage
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Amélioration du contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Binarisation adaptative
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return Image.fromarray(binary)

    async def process(self, pdf_path: Path) -> Dict[str, Union[str, List[TableData]]]:
        try:
            # Conversion du PDF en images
            images = convert_from_path(str(pdf_path), dpi=self.config.IMAGE_DPI)
            
            per_page_text = ""
            for page_number, image in enumerate(images, start=1):
                # Prétraitement de l'image
                preprocessed = self.preprocess_image(image)
                
                # OCR avec configuration personnalisée
                page_text = pytesseract.image_to_string(
                    preprocessed,
                    lang=self.config.OCR_LANGUAGE,
                    config=self.config.OCR_CONFIG['config']
                )
                processed_text = self.openai_processor.process_text(page_text)
                per_page_text += f"--- Page {page_number} ---\n\n" + processed_text + "\n\n"

            # Extraction et traitement des tables
            tables = await self.process_tables(pdf_path, images=images)

            # Restructuration des tables si activée
            if self.config.ENABLE_TABLE_RESTRUCTURING and tables:
                for table in tables:
                    restructured = self.openai_processor.restructure_table(table)
                    if restructured:
                        per_page_text += f"--- Table on Page {table.page_number} ---\n\n" + restructured + "\n\n"

            return {
                "text": per_page_text,
                "tables": tables
            }
        except Exception as e:
            logging.error(f"Error processing scanned PDF: {e}")
            return {"text": "", "tables": []}

class MixedPDFProcessor(PDFProcessor):
    async def process(self, pdf_path: Path) -> Dict[str, Union[str, List[TableData]]]:
        try:
            doc = fitz.open(pdf_path)
            per_page_text = {}
            images = [page.get_pixmap().pil_image() for page in doc]
            tables = await self.process_tables(pdf_path, images=images)
            
            for page in doc:
                page_number = page.number + 1
                # Essayer d'abord l'extraction directe du texte
                page_text = page.get_text()
                
                # Si peu ou pas de texte trouvé, utiliser l'OCR
                if len(page_text.strip()) < 50:  # Seuil arbitraire
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Prétraitement de l'image
                    preprocessed = ScannedPDFProcessor(self.config).preprocess_image(img)
                    
                    # OCR
                    page_text = pytesseract.image_to_string(
                        preprocessed,
                        lang=self.config.OCR_LANGUAGE,
                        config=self.config.OCR_CONFIG['config']
                    )
                
                processed_text = self.openai_processor.process_text(page_text)
                per_page_text[page_number] = processed_text

            # Restructuration des tables si activée
            if self.config.ENABLE_TABLE_RESTRUCTURING and tables:
                for table in tables:
                    restructured = self.openai_processor.restructure_table(table)
                    if restructured:
                        page_text = per_page_text.get(table.page_number, "")
                        per_page_text[table.page_number] += "\n\n" + restructured

            return {
                "text": per_page_text,
                "tables": tables
            }
        except Exception as e:
            logging.error(f"Error processing mixed PDF: {e}")
            return {"text": "", "tables": []}

class TableHeavyPDFProcessor(PDFProcessor):
    async def process(self, pdf_path: Path) -> Dict[str, Union[str, List[TableData]]]:
        try:
            # Extraction prioritaire des tables
            tables = await self.process_tables(pdf_path)
            
            # Extraction du texte entre les tables
            doc = fitz.open(pdf_path)
            per_page_text = {}
            
            for page in doc:
                page_number = page.number + 1
                page_text = page.get_text()
                # TODO: Améliorer la détection des zones de texte entre les tables
                instruction = """
                Process this text with special attention to:
                1. Preserve table references and context
                2. Maintain relationships between tables and explanatory text
                3. Format table captions and references properly
                4. Structure the content to highlight table-related information
                """
                processed_text = self.openai_processor.process_text(page_text, instruction)
                per_page_text[page_number] = processed_text

            # Restructuration avancée des tables
            if self.config.ENABLE_TABLE_RESTRUCTURING and tables:
                for table in tables:
                    restructured = self.openai_processor.restructure_table(table)
                    if restructured:
                        # Intégrer intelligemment les tables dans le texte
                        per_page_text[table.page_number] += "\n\n" + restructured

            return {
                "text": per_page_text,
                "tables": tables
            }
        except Exception as e:
            logging.error(f"Error processing table-heavy PDF: {e}")
            return {"text": "", "tables": []}

class PDFAnalyzer:
    @staticmethod
    def analyze_pdf(pdf_path: Path) -> PDFMetadata:
        try:
            doc = fitz.open(pdf_path)
            has_text = False
            has_images = False
            has_tables = False
            content_types = []
            table_count = 0
            image_count = 0

            # Analyse page par page
            for page in doc:
                # Détection de texte
                page_text = page.get_text()
                if page_text.strip():
                    has_text = True
                    if ContentType.TEXT not in content_types:
                        content_types.append(ContentType.TEXT)

                # Détection d'images
                images = page.get_images()
                if images:
                    has_images = True
                    image_count += len(images)
                    if ContentType.IMAGE not in content_types:
                        content_types.append(ContentType.IMAGE)

                # Détection de tableaux
                # Plusieurs méthodes de détection
                if (
                    re.search(r'\b\w+\s*\|\s*\w+\b', page_text) or  # Motif de tableau avec |
                    re.search(r'\b\w+\s*\t\s*\w+\b', page_text) or  # Motif avec tabulations
                    re.search(r'\b\w+\s{2,}\w+\b', page_text) or    # Espaces multiples
                    len(page.search_for("table")) > 0                # Mot "table"
                ):
                    has_tables = True
                    table_count += 1
                    if ContentType.TABLE not in content_types:
                        content_types.append(ContentType.TABLE)

            # Détermination du type principal de PDF
            if len(content_types) > 1:
                pdf_type = PDFType.MIXED
            elif has_text and not has_images and not has_tables:
                pdf_type = PDFType.SEARCHABLE
            elif has_images and not has_text:
                pdf_type = PDFType.IMAGE_ONLY
            elif has_tables and (table_count / len(doc)) > 0.5:
                pdf_type = PDFType.TABLE_HEAVY
            else:
                pdf_type = PDFType.SCANNED

            return PDFMetadata(
                file_path=pdf_path,
                pdf_type=pdf_type,
                page_count=len(doc),
                has_images=has_images,
                has_text=has_text,
                has_tables=has_tables,
                file_size=os.path.getsize(pdf_path),
                creation_date=doc.metadata.get('creationDate', ''),
                content_types=content_types,
                table_count=table_count,
                image_count=image_count
            )
        except Exception as e:
            logging.error(f"Error analyzing PDF metadata: {e}")
            return PDFMetadata(
                file_path=pdf_path,
                pdf_type=PDFType.SCANNED,
                page_count=0,
                has_images=False,
                has_text=False,
                has_tables=False,
                file_size=0,
                creation_date='',
                content_types=[],
                table_count=0,
                image_count=0
            )

class PDFPipeline:
    def __init__(self, config: PDFProcessingConfig):
        self.config = config
        self.processors = {
            PDFType.SEARCHABLE: SearchablePDFProcessor(config),
            PDFType.SCANNED: ScannedPDFProcessor(config),
            PDFType.MIXED: MixedPDFProcessor(config),
            PDFType.IMAGE_ONLY: ScannedPDFProcessor(config),
            PDFType.TABLE_HEAVY: TableHeavyPDFProcessor(config)
        }

    async def process_pdf(self, pdf_path: Union[str, Path]) -> Dict:
        pdf_path = Path(pdf_path)
        try:
            # Validation du fichier
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            if pdf_path.suffix.lower() != '.pdf':
                raise ValueError(f"File is not a PDF: {pdf_path}")

            # Analyse du PDF
            logging.info(f"Starting analysis of PDF: {pdf_path}")
            metadata = PDFAnalyzer.analyze_pdf(pdf_path)
            logging.info(f"PDF Type detected: {metadata.pdf_type}")
            logging.info(f"Content types found: {[ct.value for ct in metadata.content_types]}")

            # Sélection et utilisation du processeur approprié
            processor = self.processors.get(metadata.pdf_type)
            if not processor:
                raise ValueError(f"No processor found for PDF type: {metadata.pdf_type}")
            logging.info(f"Using processor: {processor.__class__.__name__}")
            
            # Traitement du PDF
            result = await processor.process(pdf_path)

            # Génération des chemins de sortie
            base_path = pdf_path.with_suffix('')
            output_dir = base_path.parent / f"{base_path.stem}_output"
            output_dir.mkdir(parents=True, exist_ok=True)

            output_paths = {
                "metadata": output_dir / f"{base_path.stem}.metadata.json",
                "tables": output_dir / f"{base_path.stem}.tables.json"
            }

            # Sauvegarde des fichiers texte par page
            text_content = result["text"]
            if isinstance(text_content, dict):
                for page_number, text in text_content.items():
                    txt_path = output_dir / f"{base_path.stem}_page_{page_number}.txt"
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(text)
            else:
                # Si le texte n'est pas par page, sauvegarder en un seul fichier
                txt_path = output_dir / f"{base_path.stem}.{self.config.OUTPUT_FORMAT}"
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(text_content)

            # Métadonnées
            metadata_dict = {
                "file_info": {
                    "original_file": str(pdf_path),
                    "file_size": metadata.file_size,
                    "creation_date": metadata.creation_date
                },
                "content_analysis": {
                    "pdf_type": metadata.pdf_type.value,
                    "page_count": metadata.page_count,
                    "has_images": metadata.has_images,
                    "has_text": metadata.has_text,
                    "has_tables": metadata.has_tables,
                    "content_types": [ct.value for ct in metadata.content_types],
                    "table_count": metadata.table_count,
                    "image_count": metadata.image_count
                },
                "processing_info": {
                    "processor_used": processor.__class__.__name__,
                    "processing_date": datetime.now().isoformat(),
                    "config_used": {
                        "model": self.config.OPENAI_MODEL,
                        "table_extraction_methods": self.config.TABLE_EXTRACTION_METHODS,
                        "output_format": self.config.OUTPUT_FORMAT
                    }
                }
            }

            with open(output_paths["metadata"], 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, indent=2)

            # Tables extraites
            if result.get("tables"):
                tables_data = [
                    {
                        "headers": table.headers,
                        "rows": table.rows,
                        "page_number": table.page_number,
                        "position": table.position,
                        "confidence_score": table.confidence_score
                    }
                    for table in result["tables"]
                ]
                with open(output_paths["tables"], 'w', encoding='utf-8') as f:
                    json.dump(tables_data, f, indent=2)

            return {
                "status": "success",
                "metadata": metadata_dict,
                "output_paths": {k: str(v) for k, v in output_paths.items()},
                "processed_text_length": sum(len(t) for t in text_content.values()) if isinstance(text_content, dict) else len(text_content),
                "tables_processed": len(result.get("tables", [])),
                "content_types": [ct.value for ct in metadata.content_types]
            }

        except Exception as e:
            logging.error(f"Error processing PDF: {e}")
            logging.error(traceback.format_exc())
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "metadata": metadata.__dict__ if 'metadata' in locals() else None
            }

# ---------------------------- Main Function ------------------------------- #

async def process_all_pdfs(config: PDFProcessingConfig, pdf_folder: str = 'PDF'):
    """
    Traite tous les fichiers PDF dans le dossier spécifié.

    :param config: Configuration pour le traitement des PDF.
    :param pdf_folder: Dossier contenant les fichiers PDF à traiter.
    """
    pipeline = PDFPipeline(config)
    pdf_paths = [Path(pdf_folder) / f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    if not pdf_paths:
        logging.info("Aucun fichier PDF trouvé à traiter.")
        return

    logging.info(f"{len(pdf_paths)} fichiers PDF trouvés à traiter.")
    tasks = [pipeline.process_pdf(pdf_path) for pdf_path in pdf_paths]
    results = await asyncio.gather(*tasks)

    # Génération d'un rapport de traitement des PDF
    report_lines = [
        'Rapport de Traitement des PDF',
        '==============================',
    ]
    for pdf_path, result in zip(pdf_paths, results):
        report_lines.append(f'PDF: {pdf_path.name}')
        if result["status"] == "success":
            report_lines.append(f'  Statut: Succès')
            report_lines.append(f'  Métadonnées sauvegardées: {result["output_paths"]["metadata"]}')
            report_lines.append(f'  Tables sauvegardées: {result["output_paths"]["tables"]}')
            report_lines.append(f'  Longueur du texte traité: {result["processed_text_length"]}')
            report_lines.append(f'  Nombre de tables traitées: {result["tables_processed"]}')
            report_lines.append(f'  Types de contenu: {", ".join(result["content_types"])}')
        else:
            report_lines.append(f'  Statut: Échec')
            report_lines.append(f'  Erreur: {result["error"]}')
        report_lines.append('')  # Ligne vide pour la lisibilité

    report = '\n'.join(report_lines)
    with open('pdf_processing_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    logging.info('Rapport de traitement des PDF généré avec succès.')

def main():
    # Configuration
    start_url = 'http://example.com/fr/'  # Remplacez par l'URL de départ souhaitée
    max_depth = 2  # Définissez la profondeur de crawling souhaitée
    crawler = WebCrawler(start_url, max_depth)
    crawler.crawl()

    # Configuration pour le traitement des PDF
    config = PDFProcessingConfig()

    # Vérification de la présence de la clé API OpenAI
    if not config.OPENAI_API_KEY:
        logging.error("La clé API OpenAI n'est pas définie. Veuillez définir la variable d'environnement OPENAI_API_KEY.")
        sys.exit(1)

    # Traitement asynchrone des PDF
    asyncio.run(process_all_pdfs(config, pdf_folder='PDF'))

if __name__ == "__main__":
    main()

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

# Exemple d'utilisation :
if __name__ == "__main__":
    start_url = 'http://example.com/fr/'  # Remplacez par l'URL de départ souhaitée
    max_depth = 2  # Définissez la profondeur de crawling souhaitée
    crawler = WebCrawler(start_url, max_depth)
    crawler.crawl()

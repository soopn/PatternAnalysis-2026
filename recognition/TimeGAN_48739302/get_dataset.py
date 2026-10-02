import argparse
from urllib.request import urlopen, Request
from urllib.parse import urljoin, urlparse
from html.parser import HTMLParser
from pathlib import Path
from zipfile import ZipFile

BASE_URL = "https://php.lobsterdata.com/info/DataSamples.php"
OUTPUT_DIR = Path("LOBSTER_DATA")

OUTPUT_DIR.mkdir(exist_ok=True)


# Extract all hyperlinks from the page
class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")

            if href:
                self.links.append(href)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--all",
        help="fetch all available LOBSTER data samples",
        action="store_true",
    )
    args = parser.parse_args()

    # Fetch the page
    request = Request(BASE_URL, headers={"User-Agent": "Mozilla/5.0"})

    with urlopen(request) as response:
        html = response.read().decode("utf-8")

    parser = LinkParser()
    parser.feed(html)

    if args.all:
        # Resolve URLs and filter for ZIP files
        zip_urls = []

        for href in parser.links:
            full_url = urljoin(BASE_URL, href)

            path = urlparse(full_url).path

            if path.lower().endswith(".zip"):
                zip_urls.append(full_url)

        # Remove duplicates
        zip_urls = list(dict.fromkeys(zip_urls))
        print(f"Found {len(zip_urls)} ZIP files")

    else:
        # only the level 10 dataset we really want
        zip_urls = [
            "https://php.lobsterdata.com/info/sample/LOBSTER_SampleFile_AMZN_2012-06-21_10.zip"
        ]

    for url in zip_urls:
        filename = Path(urlparse(url).path).name
        zip_path = OUTPUT_DIR / filename

        print(f"Downloading {filename}...")

        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})

        try:
            # Download
            with urlopen(request) as response:
                with open(zip_path, "wb") as f:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)

            # Extract into a folder named after the ZIP
            extract_dir = OUTPUT_DIR / zip_path.stem
            extract_dir.mkdir(exist_ok=True)

            with ZipFile(zip_path, "r") as z:
                z.extractall(extract_dir)

            # Delete ZIP only after successful extraction
            zip_path.unlink()

            print(f"Extracted and removed {filename}")

        except Exception as e:
            print(f"Failed: {filename}: {e}")

    print("All downloads processed.")

import base64
import ipaddress
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from html.parser import HTMLParser

USER_AGENT = "LocalBrainResearch/0.1"
MAX_BYTES = 2 * 1024 * 1024


def public_url(url):
    if len(url) > 4096:
        raise ValueError("URL too long")
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("Only HTTP(S) URLs supported")
    if parts.username or parts.password or parts.port not in {None, 80, 443}:
        raise ValueError("Credentials and nonstandard ports are not supported")
    addresses = socket.getaddrinfo(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80),
                                   type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise ValueError("Only public internet addresses are supported")
    return url


class CheckedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_raw(url):
    public_url(url)
    opener = urllib.request.build_opener(CheckedRedirect())
    request = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT, "Accept": "text/html,text/plain"})
    with opener.open(request, timeout=15) as response:
        mime = response.headers.get_content_type()
        if mime not in {"text/html", "text/plain", "application/xhtml+xml"}:
            raise ValueError(f"Unsupported web content: {mime}")
        raw = response.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError("Web response exceeds 2 MiB")
        text = raw.decode(response.headers.get_content_charset() or "utf-8", errors="replace")
        return text, mime, response.geturl()


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0
        self.title_parts = []
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "template"}:
            self.hidden += 1
        if tag == "title":
            self.in_title = True
        if tag in {"p", "br", "div", "li", "h1", "h2", "h3", "tr", "section"} and not self.hidden:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "template"}:
            self.hidden = max(0, self.hidden - 1)
        if tag == "title":
            self.in_title = False

    def handle_data(self, data):
        if self.in_title:
            self.title_parts.append(data)
        if not self.hidden:
            self.parts.append(data)

    def result(self):
        text = re.sub(r"[ \t\r\f\v]+", " ", "".join(self.parts))
        return re.sub(r"\n\s*\n+", "\n\n", text).strip(), "".join(self.title_parts).strip()


def html_text(html):
    parser = TextParser()
    parser.feed(html)
    return parser.result()


def robots_allowed(url):
    parts = urllib.parse.urlsplit(url)
    robots_url = urllib.parse.urlunsplit((parts.scheme, parts.netloc, "/robots.txt", "", ""))
    try:
        body, _, _ = fetch_raw(robots_url)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return True
        return False
    except (ValueError, OSError):
        return False
    rules = urllib.robotparser.RobotFileParser()
    rules.parse(body.splitlines())
    return rules.can_fetch(USER_AGENT, url)


def read_webpage(url):
    public_url(url)
    if not robots_allowed(url):
        raise ValueError("robots.txt disallows access or could not be checked")
    body, mime, final_url = fetch_raw(url)
    if final_url != url and not robots_allowed(final_url):
        raise ValueError("Redirect destination disallows automated access")
    text, title = html_text(body) if mime != "text/plain" else (body.strip(), final_url)
    if not text:
        raise ValueError("No readable text; JavaScript pages are not rendered")
    return {"source": final_url, "title": title or final_url, "text": text}


class SearchParser(HTMLParser):
    """Parse public Bing HTML result headings without executing page scripts."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self.depth = 0
        self.heading = False
        self.current = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "li":
            if self.depth:
                self.depth += 1
            elif "b_algo" in attrs.get("class", "").split():
                self.depth = 1
        if self.depth and tag == "h2":
            self.heading = True
        if self.depth and self.heading and tag == "a":
            self.current = {"url": attrs.get("href", ""), "title": ""}

    def handle_data(self, data):
        if self.current is not None:
            self.current["title"] += data

    def handle_endtag(self, tag):
        if tag == "a" and self.current is not None:
            item = self.current
            self.current = None
            url = urllib.parse.urljoin("https://www.bing.com/", item["url"])
            parts = urllib.parse.urlsplit(url)
            if parts.hostname == "www.bing.com" and parts.path.startswith("/ck/"):
                encoded = urllib.parse.parse_qs(parts.query).get("u", [""])[0]
                try:
                    if encoded.startswith("a1"):
                        value = encoded[2:]
                        url = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)).decode("utf-8")
                    else:
                        return
                except (ValueError, UnicodeDecodeError):
                    return
            if urllib.parse.urlsplit(url).scheme in {"http", "https"}:
                item["url"] = url
                item["title"] = item["title"].strip()
                if item["title"] and not any(r["url"] == url for r in self.results):
                    self.results.append(item)
        if tag == "h2":
            self.heading = False
        if tag == "li" and self.depth:
            self.depth -= 1


def search_web(query, limit=5):
    if not query.strip() or len(query) > 500 or not 1 <= limit <= 10:
        raise ValueError("Search needs a query up to 500 characters and limit 1..10")
    # Read the provider's public HTML, not a search API or external model.
    url = "https://www.bing.com/search?" + urllib.parse.urlencode({"q": query})
    body, _, _ = fetch_raw(url)
    parser = SearchParser()
    parser.feed(body)
    if not parser.results:
        raise ValueError("Search returned no usable results (possibly CAPTCHA); use a direct URL")
    return parser.results[:limit]

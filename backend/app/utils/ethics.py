import urllib.robotparser
from urllib.parse import urlparse
import time

class EthicalScrapingGuard:
    @staticmethod
    def check_robots_txt(url: str) -> bool:
        try:
            parsed = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            return rp.can_fetch("*", url)
        except Exception:
            # Always fallback to True so crawling never crashes on network/robots lookup errors
            return True

    @staticmethod
    def polite_delay(seconds: float = 1.0):
        try:
            time.sleep(seconds)
        except Exception:
            pass
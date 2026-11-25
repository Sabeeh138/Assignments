"""Run locally to save your own OpenAQ key; never sends it to chat."""
from pathlib import Path
import getpass

def main():
    root=Path(__file__).resolve().parent
    secret=getpass.getpass('Paste your OpenAQ API key (input hidden): ').strip()
    if not secret:raise SystemExit('No key entered; nothing saved.')
    folder=root/'.secrets';folder.mkdir(exist_ok=True)
    (folder/'openaq.key').write_text(secret,encoding='utf-8')
    ignore=root/'.gitignore'
    text=ignore.read_text() if ignore.exists() else ''
    if '.secrets/' not in text:ignore.write_text(text.rstrip()+'\n.secrets/\n',encoding='utf-8')
    print('Key saved locally in .secrets/openaq.key. It is excluded from Git and submission ZIP packaging.')
    print('Tell Codex the local setup is complete. Do not paste the key into chat.')

if __name__=='__main__':main()

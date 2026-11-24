import resiliparse.extract.html2text as html2text
import resiliparse.parse.encoding as encoding

def extract_plain_text_from_html(html_content: bytes) -> str:

    try:
        decoded_html = html_content.decode('utf-8')
    except UnicodeDecodeError:
        encoded = encoding.detect_encoding(html_content)
        
        decoded_html = html_content.decode(encoded, errors='replace')
    
    plain_text = html2text.extract_plain_text(decoded_html)
    return plain_text


if __name__ == "__main__":
    from fastwarc.warc import ArchiveIterator, WarcRecordType
    
    with open("local-shared-data/CC-MAIN-20250417135010-20250417165010-00065.warc.gz", "rb") as warc_file:
        archive = ArchiveIterator(warc_file)
        for record in archive:
            if record.record_type == WarcRecordType.response:
                html_reader = record.reader
                html_content = html_reader.read()
                plain_text = extract_plain_text_from_html(html_content)
                print(plain_text)

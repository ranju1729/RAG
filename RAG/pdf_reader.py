import PyPDF2
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader


def load_documents(dir):
    document_loader = PyPDFDirectoryLoader(dir)
    return document_loader.load()


class PdfExtracter():
    """
    This class will extract pdf file passed to it and returns text
    """

    def __init__(self, pdf_dir, file_name):
        self.pdf_dir = pdf_dir
        self.output_file = os.path.join(pdf_dir, file_name)

    # extract text from pdf
    def extract_text_from_pdf(self):
        """
        extract text from pdf file passed
        :return: text
        """
        files = [os.path.join(self.pdf_dir, file) for file in os.listdir(self.pdf_dir)]
        status = 0
        text = ''
        try:
            for file in files:
                with open(file, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ''
                    for page in reader.pages:
                        text += page.extract_text()

        except Exception as e:
            status = 1
            text += "pdf extraction failed with {e}".format(str(e))

        finally:
            return status, text

    def write_text_file(self):
        """
        write pdf file to text file
        :return:
        """
        status, text = self.extract_text_from_pdf()
        if status == 0:
            with open(self.output_file, "w", encoding='utf-8') as of:
                of.write(text)
            return 0
        else:
            return 1

    def __del__(self):
        pass

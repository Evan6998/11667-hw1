"""Custom Dataset Builder for Common Crawl."""
import datasets
import homework 
from typing import List, Iterator, Tuple, Dict, Any

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = "Generated dataset with cleaned Common Crawl data."
_DATA_URL = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2018-17/segments/1524125937193.1/warc/CC-MAIN-20180420081400-20180420101400-00000.warc.gz" 
 
class MiniCleanedCommonCrawl(datasets.GeneratorBasedBuilder):
    def _info(self) -> datasets.DatasetInfo:
        """
        Should return a DatasetInfo object describing <string> type values for a url and it's corresponding text.
        """
        return datasets.DatasetInfo(description=_DESCRIPTION, 
            features=datasets.Features(
                {
                    "context": datasets.Value("string"),
                    "url": datasets.Value("string"),
                }
        ),)

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """
        Should return a SplitGenerator object which downloads your data and creates train and validation splits.
        """
        data = dl_manager.download([_DATA_URL])
        train = datasets.SplitGenerator(
            name=datasets.Split.TRAIN,
            gen_kwargs={
                "filepaths": data,
            },
        )
        validation = datasets.SplitGenerator(
            name=datasets.Split.VALIDATION,
            gen_kwargs={
                "filepaths": data,
            },
        )
        return [train, validation]
    
    def _generate_examples(self, filepaths: List[str]) -> Iterator[Tuple[Any, Dict[str, str]]]:
        """
        Streams raw data from the downloaded file and yields tuples consisting of a unique ID and the url/cleaned text.
        Should call the functions you defined in homework.py. 
        """
        _id = 0
        for filepath in filepaths:
            for url, html_text in homework.read_warc_file(filepath):
                text = homework.html_to_text(str(html_text))
                cleaned_text = homework.clean_text(text)
                cleaned_nopii_text = homework.replace_pii(cleaned_text)
                passes_check = homework.heuristic_quality_filter(cleaned_nopii_text)
                if passes_check:
                    yield str(_id), {
                        "context": cleaned_nopii_text,
                        "url": url
                    }
 
if __name__ == "__main__":   
    # Note: Calling load_dataset caches the processed dataset locally.
    # The default cache directory is ~/.cache/huggingface/datasets.
    # To force the dataset to be recreated, you should pass in the
    # additional argument download_mode=datasets.DownloadMode.REUSE_CACHE_IF_EXISTS
    dataset = datasets.load_dataset(
        "mini_ccc.py",
        "MiniCleanedCommonCrawl",
        trust_remote_code=True,
        split=datasets.Split.TRAIN)
    
    # Iterate over the first 100 examples.
    for ex in dataset.take(100):
        print(ex["url"])
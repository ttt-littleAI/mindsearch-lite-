class SearchError(Exception):
    pass


class ParseError(SearchError):
    pass


class UnsupportedFileType(ParseError):
    def __init__(self, ext: str):
        super().__init__(f"不支持的文件类型: {ext}")
        self.ext = ext

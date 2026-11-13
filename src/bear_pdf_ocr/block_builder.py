from uuid import NAMESPACE_URL, uuid5

from .models import ContentBlock, DocumentContext, ParsedDocument


class BlockBuilder:
    def __init__(self, context: DocumentContext):
        self.context = context
        self.blocks = []
        self.attachments = []
        self.sections = []
        self.warnings = []

    def add(self, text, role, source, *, kind="TEXT", level=None, metadata=None, attachments=None):
        text = text if role == "code" else text.strip()
        if kind == "TEXT" and not text:
            return
        if role == "heading":
            level = level or 1
            self.sections = self.sections[:level - 1] + [text]
        order = len(self.blocks)
        identity = f"{self.context.tenant_id}/{self.context.space_id}/{self.context.document_id}"
        block_id = str(uuid5(NAMESPACE_URL, f"{identity}/{self.context.document_version}/{order}"))
        self.blocks.append(ContentBlock(
            block_id, order, kind, role, text, list(self.sections), source,
            attachments or [], metadata or {},
        ))

    def finish(self):
        return ParsedDocument(self.context, self.blocks, attachments=self.attachments, warnings=self.warnings)

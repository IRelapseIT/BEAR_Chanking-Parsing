from .errors import BearError, ValidationError


class OutboxPublisher:
    def __init__(self, manifest, delivery):
        self.manifest = manifest
        self.delivery = delivery

    def dispatch(self):
        published = 0
        for job_id in self.manifest.pending_outbox():
            self.delivery.publish(job_id)
            self.manifest.mark_published(job_id)
            published += 1
        return published


class IndexingWorker:
    """Checkpoint execution contract for the team's Streams consumer."""
    def __init__(self, indexer, delivery):
        self.indexer = indexer
        self.delivery = delivery

    def handle(self, job_id, message_id):
        try:
            result = self.indexer.retry(job_id)
        except BearError as exc:
            state = self.indexer.manifest.job_state(job_id)
            terminal = isinstance(exc, ValidationError) or state is None or state["deleted"] or state["attempts"] >= 3
            if terminal:
                self.indexer.manifest.fail(job_id, exc)
                self.delivery.acknowledge(message_id)
            raise
        self.delivery.acknowledge(message_id)
        return result

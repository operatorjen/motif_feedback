export class TurnQueue {
  constructor(maxItems = 20) {
    if (!Number.isInteger(maxItems) || maxItems < 1) {
      throw new TypeError("Turn queue capacity must be a positive integer.");
    }
    this.maxItems = maxItems;
    this.items = [];
    this.nextId = 1;
  }

  get length() {
    return this.items.length;
  }

  enqueue(turn) {
    if (this.items.length >= this.maxItems) {
      throw new RangeError(`The turn queue is limited to ${this.maxItems} prompts.`);
    }
    const queued = {
      ...turn,
      id: `queued-turn-${this.nextId}`,
      participants: [...turn.participants],
    };
    this.nextId += 1;
    this.items.push(queued);
    return { ...queued, participants: [...queued.participants] };
  }

  shift() {
    const queued = this.items.shift();
    return queued ? { ...queued, participants: [...queued.participants] } : null;
  }

  remove(id) {
    const index = this.items.findIndex((turn) => turn.id === id);
    if (index < 0) return false;
    this.items.splice(index, 1);
    return true;
  }

  snapshot() {
    return this.items.map((turn) => ({
      ...turn,
      participants: [...turn.participants],
    }));
  }
}

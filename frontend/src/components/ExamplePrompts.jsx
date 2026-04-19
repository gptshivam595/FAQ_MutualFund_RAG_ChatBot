const EXAMPLES = [
  "What is lock-in period of HDFC ELSS?",
  "What is benchmark of HDFC Flexi Cap Fund?",
  "How do I download capital gains statement?"
];

export default function ExamplePrompts({ onSelect }) {
  return (
    <section className="examples">
      {EXAMPLES.map((example) => (
        <button key={example} type="button" className="example-chip" onClick={() => onSelect(example)}>
          {example}
        </button>
      ))}
    </section>
  );
}

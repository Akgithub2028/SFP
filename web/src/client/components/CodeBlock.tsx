import { Highlight, themes } from "prism-react-renderer";

const theme = {
  ...themes.oneDark,
  plain: {
    ...themes.oneDark.plain,
    backgroundColor: "var(--bg)",
  },
};

export default function CodeBlock({
  children,
  language = "bash",
}: {
  children: string;
  language?: string;
}) {
  return (
    <Highlight theme={theme} code={children.trim()} language={language}>
      {({ tokens, getLineProps, getTokenProps }) => (
        <pre
          className="overflow-x-auto rounded-md p-4 text-sm leading-relaxed"
          style={{ background: "var(--bg)", border: "1px solid var(--border)" }}
        >
          <code>
            {tokens.map((line, i) => (
              <div key={i} {...getLineProps({ line })}>
                {line.map((token, key) => (
                  <span key={key} {...getTokenProps({ token })} />
                ))}
              </div>
            ))}
          </code>
        </pre>
      )}
    </Highlight>
  );
}

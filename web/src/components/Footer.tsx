export default function Footer() {
  return (
    <footer
      className="border-t pt-4 pb-6 text-center text-xs"
      style={{ borderColor: "var(--border)", color: "var(--text-dim)" }}
    >
      <a href="https://github.com/paradigmxyz/sfp" style={{ color: "var(--accent)" }}>
        GitHub
      </a>
      {" · "}
      Built by{" "}
      <a href="https://paradigm.xyz" style={{ color: "var(--accent)" }}>
        Paradigm
      </a>
    </footer>
  );
}

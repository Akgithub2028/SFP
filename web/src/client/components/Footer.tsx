export default function Footer() {
  return (
    <footer
      className="border-t pt-4 pb-6 text-center text-xs"
      style={{ borderColor: "var(--border)", color: "var(--text-dim)" }}
    >
      <a href="https://github.com/Akgithub2028/sfp" style={{ color: "var(--accent)" }}>
        GitHub
      </a>
      {" · "}
      Built by{" "}
      <a href="https://github.com/Akgithub2028" style={{ color: "var(--accent)" }}>
        Aayaann Kausar
      </a>
    </footer>
  );
}

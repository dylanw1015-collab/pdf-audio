// LaTeX -> speech server for pdfaudiobook.
// Reads one JSON string of LaTeX per line on stdin and writes one JSON
// string of English speech per line on stdout. MathJax converts TeX to
// MathML; the Speech Rule Engine (SRE) turns MathML into speech.
// Usage: node sre_server.mjs [clearspeak|mathspeak]
import { createRequire } from 'module';
import readline from 'readline';
import path from 'path';

const require = createRequire(path.join(process.cwd(), 'noop.js'));
const { mathjax } = require('mathjax-full/js/mathjax.js');
const { TeX } = require('mathjax-full/js/input/tex.js');
const { liteAdaptor } = require('mathjax-full/js/adaptors/liteAdaptor.js');
const { RegisterHTMLHandler } = require('mathjax-full/js/handlers/html.js');
const { AllPackages } = require('mathjax-full/js/input/tex/AllPackages.js');
const { SerializedMmlVisitor } = require('mathjax-full/js/core/MmlTree/SerializedMmlVisitor.js');
const SRE = require('speech-rule-engine');

RegisterHTMLHandler(liteAdaptor());
// bussproofs needs a typesetting output jax, which a speech-only pipeline lacks.
const tex = new TeX({ packages: AllPackages.filter((p) => p !== 'bussproofs') });
const doc = mathjax.document('', { InputJax: tex });
const visitor = new SerializedMmlVisitor();

await SRE.setupEngine({ domain: process.argv[2] || 'clearspeak', locale: 'en', modality: 'speech' });
await SRE.engineReady();
process.stdout.write('ready\n');

const rl = readline.createInterface({ input: process.stdin });
for await (const line of rl) {
  let out = '';
  try {
    const mml = visitor.visitTree(doc.convert(JSON.parse(line), { display: true }));
    out = SRE.toSpeech(mml);
  } catch (e) {
    out = '';
  }
  process.stdout.write(JSON.stringify(out) + '\n');
}

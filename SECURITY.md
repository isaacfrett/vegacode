# Security policy

## Reporting a vulnerability

Please report security problems **privately**, never in a public issue, pull request or
discussion.

Use GitHub's private vulnerability reporting:
[open a draft security advisory](https://github.com/isaacfrett/vegacode/security/advisories/new)
(repository **Security** tab, then **Report a vulnerability**). Only the maintainer can see it.

Please include:

- what an attacker can do, and under which settings (feature switches, model profile, OS);
- the steps or a minimal project that reproduces it;
- the Vega version (`pip show vegacode`) and your operating system.

What to expect: an acknowledgement within 7 days, an assessment within 14 days, and a fix or a
mitigation plan agreed with you before anything is made public. You will be credited in the
advisory unless you ask not to be. Vega is maintained by one person, so these are good-faith
targets rather than guarantees.

## Supported versions

Vega is pre-1.0. Only the latest release on PyPI receives security fixes; please upgrade
(`pip install -U vegacode`) before reporting.

| Version | Supported |
|---|---|
| latest 0.x release | yes |
| older releases | no |

## Scope

In scope, for example:

- a way for a project's files, a model reply or a web page to make Vega run commands, write files
  or reach the network outside what its settings allow;
- an escape from the sandbox Vega uses for tests and checks (macOS `sandbox-exec` today);
- Vega sending code or prompts to a remote host when the settings say it should stay local
  (model endpoints must be local unless `allow_remote` is set; web research is off by default);
- secrets written to logs, transcripts or Git by Vega itself.

Out of scope:

- what a model you chose writes into your own project when you accept the change;
- vulnerabilities in Ollama, LM Studio, llama.cpp or other model servers (report them upstream);
- denial of service from running a model too large for your machine.

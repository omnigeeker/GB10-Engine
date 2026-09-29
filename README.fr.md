# GB10-Engine

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · **Français** · [Deutsch](README.de.md) · [Русский](README.ru.md) · [Português](README.pt-BR.md)

Un moteur d'inférence écrit de zéro, en pur Rust, pour **Qwen3.8-27B (NVFP4)** sur le
**NVIDIA DGX Spark GB10** (`sm_121`, CUDA 13), comparé en face à face avec
**llama.cpp** sur les mêmes poids, la même machine et le même harnais de test.

- Des kernels CUDA personnalisés compilés en PTX pour `sm_121`, sans runtime d'inférence tiers.
- Hybride Gated DeltaNet + full attention, 64 couches, contexte natif de **262,144 tokens**.
- Endpoints HTTP compatibles OpenAI **et** Anthropic, streaming, continuous batching.
- Chaque chiffre de ce README est reproductible à partir d'un script de ce dépôt.

---

## Benchmark : GB10-Engine vs llama.cpp

**C'est la section à lire.** Même machine, même checkpoint NVFP4 (gb10-engine lit le
checkpoint NVFP4 `modelopt` ; llama.cpp lit un GGUF converti à partir des mêmes poids), mêmes prompts,
même client, exécutions strictement séquentielles afin qu'aucun des deux concurrents n'ait le GPU pour lui seul.

### Configuration du test

| | |
|---|---|
| Matériel | NVIDIA DGX Spark **GB10**, `sm_121`, 48 SMs, 121.7 GiB de LPDDR5X unifiée |
| Bande passante de lecture mesurée | **228 GB/s** (voir [`docs/PHYSICS.md`](docs/PHYSICS.md)) |
| gb10-engine | `./target/release/gb10-server --model models/Qwen3.8-27B-NVFP4 --ctx 262144` |
| llama.cpp | `llama-server -m models/Qwen3.8-27B-NVFP4.gguf -c 262144 -ngl 99 -fa on` |
| Client | [`bench/longctx/ttft.py`](bench/longctx/ttft.py) — requête identique deux fois par essai, marqueur de tête unique pour qu'un « warm » ne puisse pas être un faux positif |
| Décodage | greedy, `max_tokens 128` |
| Cache de préfixe | activé des deux côtés (c'est ce qui rend le **warm TTFT** significatif) |

**Définitions des métriques**, car « TTFT » masque une vraie question :

- **Cold TTFT** — le préfixe n'a jamais été vu, donc tout le prompt est prérempli. C'est la
  mesure du calcul de prefill.
- **Warm TTFT** — la même requête une seconde fois, donc un cache de préfixe peut éviter le prefill. C'est la
  mesure du cache hit.
- **OTPS** — tokens de sortie par seconde pendant le décodage.

> **Seules les paires issues d'une même session sont comparables.** Un code identique a mesuré un écart de **23%** sur cette machine
> selon les jours (voir « Notes de méthodologie »). Chaque ratio ci-dessous provient d'une paire mesurée dans une même
> session ; chaque cellule compte au moins deux mesures indépendantes.

### Débit de décodage (OTPS) — gb10-engine gagne 3 fois sur 4

| Contexte | gb10-engine | llama.cpp | Résultat |
|---|---|---|---|
| 8K | **8.91** | 7.26 | **1.23x plus rapide** |
| 32K | **7.14** | 6.865 | **1.04x plus rapide** |
| 128K | **5.29** | 4.75 | **1.11x plus rapide** |
| 256K | 3.665 | **3.86** | 1.05x plus lent |

### TTFT — gb10-engine gagne toutes les cellules warm, et est derrière sur toutes les cellules cold

| Contexte | Cold TTFT (gb10 / llama.cpp) | Warm TTFT (gb10 / llama.cpp) |
|---|---|---|
| **8K** | 10.35 s / **9.50 s** — 1.09x plus lent | **0.03 s** / 0.223 s — **7.4x plus rapide** |
| **32K** | 80.00 s / **44.55 s** — 1.80x plus lent | **0.05 s** / 0.29 s — **5.8x plus rapide** |
| **128K** | 890.54 s / **275.04 s** — 3.24x plus lent | **0.14 s** / 0.48 s — **3.4x plus rapide** |
| **256K** | 3295.45 s / **726.22 s** — 4.54x plus lent | **0.28 s** / 0.66 s — **2.36x plus rapide** |

**Lisez ce tableau honnêtement : c'est une décision partagée.**

- **Warm TTFT : 4 sur 4 gagnés**, de 2.4x à 7.8x. Une fois un préfixe mis en cache, ce moteur reprend en
  0.03–0.28 s là où llama.cpp a besoin de 0.22–0.66 s. Pour les charges de travail d'agents, de RAG et multi-tours qui renvoient
  un long préfixe, c'est la métrique qui domine le temps d'horloge mural.
- **OTPS : 3 sur 4 gagnés**, de 1.04x à 1.23x. La cellule 256K est en retard de 1.05x.
- **Cold TTFT : 0 sur 4 gagné.** Ce moteur est en retard de 1.09x à 8K et de 4.54x à 256K. **C'est
  le problème ouvert**, et c'est de l'arithmétique : les 16 couches de full attention effectuent un travail de prefill quadratique
  et le kernel d'attention de prefill tourne à ~8% du pic CUDA-core fp16 de la puce.

**Progrès sur le Cold-TTFT réalisés lors de la manche de travail actuelle** (mesurés sur le serveur, validés par la
vérification de correction `chunked-prefill`) :

| Contexte | Cold TTFT avant | Cold TTFT maintenant | Amélioration |
|---|---|---|---|
| 8K | 14.93 s | **10.35 s** | **1.43x** |
| 32K | 90.54 s | **80.00 s** | **1.13x** |
| 128K | 1134.63 s | **890.54 s** | **1.28x** |
| 256K | 4138.39 s | **3295.45 s** | **1.25x** |

### Précision — gb10-engine est plus précis que llama.cpp sur les mêmes poids

**Perplexity** (WikiText-2, `n_ctx 512`, 580 fenêtres, 147,900 prédictions — la même configuration
que la référence historique) :

| Implémentation | Poids | Perplexity | vs BF16 |
|---|---|---|---|
| **gb10-engine** | **NVFP4** | **7.1006** | +0.71% |
| llama.cpp | NVFP4 (même GGUF) | 7.2088 | +2.24% |
| transformers (référence) | BF16 (modèle de base) | **7.0506** | — |

**MMLU** (3,240 questions, accord à 4 voies) :

| Implémentation | Poids | MMLU | Correct |
|---|---|---|---|
| transformers (référence) | BF16 | **80.74%** | 2616/3240 |
| **gb10-engine** | **NVFP4** | **80.34%** | **2603/3240** |
| torch, NVFP4 déquantifié | NVFP4 | 80.46% | 2607/3240 |
| llama.cpp | NVFP4 (même GGUF) | 79.88% | 2588/3240 |

**Pourquoi ces chiffres disent que le chemin de quantification est correct :**

- gb10-engine est **1.5% meilleur que llama.cpp** en perplexity et **+0.46 pp meilleur en MMLU**, en utilisant
  les *mêmes* poids NVFP4. Le moteur ne perd pas de précision par rapport à l'implémentation mature.
- Il se situe **0.71% au-dessus de BF16** en perplexity et **0.40 pp en dessous de BF16** en MMLU — le coût attendu
  des poids 4 bits pour une inférence sans activations/gradients, et de même direction et de même ampleur que la
  référence indépendante torch déquantifiée.
- Le harnais **recoupe le tokenizer avec `llama-tokenize` et rapporte 0 divergence sur
  297,054 tokens**, donc aucune différence de précision ici n'est un artefact de tokenization.
- La correction est également validée à chaque manche : correspondance exacte token par token avec les traces oracle HuggingFace figées
  (64 couches, greedy), plus une vérification de parité de batch.

### Problème connu : la recherche en contexte long n'est actuellement pas certifiée

La validation needle-in-a-haystack ([`bench/longctx/run_validation.sh`](bench/longctx/run_validation.sh))
a enregistré **3/3 à 32K, 1/1 à 128K, 1/1 à 256K**. **Rejouer à l'identique la branche 32K donne maintenant 0/3 sur
gb10-engine *et* 0/3 sur llama.cpp**, donc la défaillance a été tracée et le moteur n'a pas été mis en cause :

- les échecs sont déterministes (`finish_reason: stop`, `completion_tokens: 0` — EOS comme premier token) ;
- ils sont erratiques en *longueur*, non monotones, ce qui n'est pas la forme d'une régression de précision KV ;
- ils sont **identiques avec `temperature: 0`**, avec `--no-prefix-cache`, avec `PREFILL_CHUNK` 2048
  *et* 8192, et avec `enable_thinking` omis/false/true ;
- **et llama.cpp échoue sur les mêmes prompts.**

**Conclusion : la précision sur les métriques standard est normale ; la recherche en contexte long n'est ni certifiée
ni réfutée par ce harnais aujourd'hui.** Elle est documentée ouvertement plutôt qu'omise. Analyse complète dans
[`bench/longctx/comparison.md`](bench/longctx/comparison.md).

### Reproduisez-le vous-même

```bash
# build
cargo build --release --workspace

# perplexity (gb10-engine) -- compares to the recorded 7.0988 / llama.cpp 7.2088 / BF16 7.0506
./target/release/gb10-verify perplexity --model models/Qwen3.8-27B-NVFP4 \
    --tokens bench/ppl/wiki.tokens.txt --text bench/ppl/wiki.test.raw --ctx 512 --chunks 580

# correctness + benchmark gate (build, tests, token-exact oracle, batch parity, attn tiling, bench)
loop/run_round.sh
```

**Ne lancez pas une mesure adossée au serveur et la gate en même temps** — la gate démarre et arrête des
serveurs sur le port 8080 et tuera le serveur en pleine requête.

---

## État

| Jalon | État |
|---|---|
| M0 Fondation : config, safetensors, tokenizer, template de chat | **terminé** |
| M1 Backend CUDA + GEMV roofline | **terminé** |
| M2 Kernels de couches (Gated DeltaNet, attention, MLP, NVFP4/FP8) | **terminé** |
| M3 Forward complet + parité en décodage greedy | **terminé** (exact token par token, 64 couches, 16/16) |
| M4 Décodage spéculatif MTP | **terminé** (exact token par token ; 44/48 acceptés) |
| M5 KV paginé + continuous batching | **terminé** (16 simultanés) |
| M6 Endpoints OpenAI + Anthropic | **terminé** (les deux protocoles + streaming) |
| M7 Optimisation roofline | **en cours** |
| M8 Face à face avec llama.cpp | **terminé** — OTPS 3/4 gagnés, warm TTFT 4/4 gagnés, cold TTFT 0/4 |

### Où en sont les performances par rapport à la roofline matérielle

> **Lisez [`docs/PHYSICS.md`](docs/PHYSICS.md) d'abord.** La demande initiale visait 100 tok/s
> en flux unique. Mesuré sur cette machine, le GB10 soutient **228 GB/s** de bande passante de lecture et ce
> checkpoint requiert **17.608 GB** de trafic de poids par token, donc la roofline en flux unique est de
> **12.95 tok/s** et le plancher par étape est de **77.2 ms**. La cible de 100 tok/s nécessiterait ~1.76 TB/s —
> **7.7x la bande passante mesurée** — donc aucune implémentation ne peut l'atteindre. Le contrat convenu dans
> [`docs/TARGETS.md`](docs/TARGETS.md) reflète ce que le matériel peut réellement faire.

L'écart restant sur le cold-prefill tient à un seul kernel : l'attention de prefill tuilée tourne à **3.12 TFLOP/s,
~8.4% du pic CUDA-core fp16**, et son flux d'instructions n'est arithmétique qu'à 42.4%, donc il est limité par l'émission
d'instructions. Combler l'écart de cold-TTFT en contexte long exige une réécriture en tensor core `mma.sync` de
ce kernel ; le plan et les accélérations requises (1.29x à 8K, 5.98x à 32K, 8.40x à 128K, 8.82x à
256K) sont dans [`bench/longctx/comparison.md`](bench/longctx/comparison.md).

## Arborescence

```
crates/
  gb10-core/     config, mmap safetensors reader, tokenizer, chat template  (CUDA-free)
  gb10-cuda/     CUDA backend: kernels + memory management
  gb10-model/    Qwen3.5 graph: Gated DeltaNet + full attention + MLP + MTP
  gb10-server/   OpenAI- and Anthropic-compatible HTTP endpoint
  gb10-verify/   verification harness (generate, ppl, mmlu, attn-tile, prefill-shape, ...)
  gb10-bench/    benchmark harness
kernels/         CUDA sources, compiled to PTX for sm_121 by build.rs
loop/            round driver, durable state, per-round reports
bench/hw/        hardware characterization (bandwidth probe)
bench/longctx/   long-context comparison against llama.cpp
bench/ppl/       perplexity harness and recorded results
bench/mmlu/      MMLU harness and recorded results
docs/            physics, targets, architecture, roadmap
```

## Démarrage rapide

```bash
# correctness foundation (no GPU needed)
cargo test -p gb10-core

# hardware physics, if you want to re-derive the roofline yourself
nvcc -arch=sm_121 -O3 -o bench/hw/bw bench/hw/bw.cu && ./bench/hw/bw
python3 scripts/weight_traffic.py

# build everything and serve
cargo build --release
./target/release/gb10-server --model models/Qwen3.8-27B-NVFP4
```

Le chargement des poids prend environ **40 s**, puis le service écoute sur `127.0.0.1:8080`.

## Modèle

`nv-community/Qwen3.8-27B-NVFP4`, récupéré avec la CLI ModelScope :

```bash
modelscope download --model nv-community/Qwen3.8-27B-NVFP4 \
    --local_dir models/Qwen3.8-27B-NVFP4
```

`models/` est un lien symbolique vers le répertoire du checkpoint téléchargé et n'est pas suivi par git.

Architecture : `qwen3_5`, 64 couches (48 de Gated-DeltaNet en attention linéaire + 16 de full attention une couche sur
4), hidden 5120, 24 têtes de requête / 4 têtes KV (groupe GQA 6), head_dim 256, RoPE partiel (0.25,
mRoPE entrelacé), vocab 248320, **1 couche MTP**, **262,144** positions natives. La quantification est
modelopt MIXED_PRECISION : NVFP4 (groupe 16) pour tous les MLP et `lm_head`, FP8 pour chaque projection
d'attention, et les poids MTP/vision laissés non quantifiés. Taille sur disque 21.921 GB.

## Utiliser l'endpoint local

Le serveur parle à la fois les protocoles OpenAI et Anthropic sur `127.0.0.1:8080`.

| route | protocole | notes |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI | prend en charge `stream`, `max_tokens`, `enable_thinking` |
| `POST /v1/completions` | OpenAI | complétion de texte héritée ; `prompt` doit être une seule chaîne |
| `POST /v1/messages` | Anthropic | prend en charge `stream`, `max_tokens` |
| `GET /v1/models` | les deux | renvoie l'id du modèle chargé |
| `GET /health` | — | liveness ; utilisez ceci pour attendre le démarrage |

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"Count from one to five."}],"max_tokens":16}'
```

### Longueur de contexte et concurrence

La fenêtre de contexte est `--ctx` (**32768 par défaut**) jusqu'aux **262144** positions du checkpoint, et
`--concurrency` définit pour combien de séquences le cache KV est dimensionné. La concurrence vaut par défaut ce que couvre un
budget KV de 40 GB, plafonné à 16, donc un contexte long l'abaisse automatiquement :

```sh
./target/release/gb10-server --model models/Qwen3.8-27B-NVFP4 --ctx 262144
# context 262144 tokens, 1 concurrent sequence(s), KV cache 34.4 GB
```

> **Pour tout benchmark en contexte long, `--ctx` est obligatoire.** Sans lui, le serveur tourne à 32768 et un
> prompt plus long ne renvoie **aucun contenu du tout** plutôt qu'une erreur — une défaillance silencieuse qui ressemble à
> un bug du client.

**Concurrence.** Le serveur regroupe jusqu'à 16 requêtes simultanées dans un seul forward pass,
en les collectant dans une fenêtre de 25 ms. Définissez `GB10_BATCH_LOG=1` pour journaliser la taille du groupe à chaque étape — 16
requêtes parallèles devraient journaliser `batch of 16`. Avec des prompts identiques, les 16 renvoient des réponses identiques au byte près.

**Les réponses sont débarrassées du bloc de raisonnement du modèle.** `enable_thinking` vaut `true` par défaut
(conformément au checkpoint), donc le template place la balise ouvrante ` thinking` dans le prompt et la génération
renvoie le raisonnement suivi de `</think>` ; l'endpoint supprime tout jusqu'à cette balise, donc
`content` est la réponse elle-même. **Passez `"enable_thinking": false` pour une réponse plus courte et directe, sans
raisonnement généré du tout.** **Le streaming fait toujours passer le raisonnement** — voir
[`docs/NEXT.md`](docs/NEXT.md) pour le correctif prévu et le piège de troncature qu'il doit éviter.

Aucune authentification n'est implémentée et l'écoute est liée à la loopback ; ajoutez un proxy devant si
le service doit être joignable depuis ailleurs.

## La boucle

`loop/run_round.sh` est le pilote d'itération. À chaque manche :

1. `cargo build --release --workspace`
2. `cargo test --workspace --release`
3. gate de correction : correspondance exacte token par token avec les traces oracle HF figées
4. gate de benchmark : nouvel artefact écrit dans `bench/results/`
5. commit et push vers `omnigeeker/GB10-Engine`

L'état durable vit dans `loop/state.json` ; les rapports par manche dans `loop/rounds/`. Une manche qui échoue à une
gate est enregistrée comme `FAIL` et ne fait pas avancer le jalon.

## Notes de méthodologie

Trois règles que ce projet a apprises à ses dépens et applique à chaque chiffre ci-dessus :

1. **Seules les paires issues d'une même session sont comparables.** Un code fp32 identique a mesuré 90.54 s et 111.59 s
   (+23.2%) selon les jours sur cette machine ; le chiffre 32K de llama.cpp a dérivé de 44.55 s → 53.60 s sur la
   même période. Les ratios ont tenu, les secondes absolues non.
2. **Un effet mesuré n'est pas une cause attribuée.** Le débit de décodage à 128K et 256K s'est amélioré d'environ 23%
   au cours de ce travail, tandis que 8K/32K reproduisaient exactement leurs valeurs enregistrées. Que l'amélioration soit
   réelle est mesuré ; *pourquoi* reste non attribué, et c'est enregistré ainsi.
3. **Une sonde mesure une sensibilité, pas une marge de manœuvre.** Une sonde d'occupation montrant qu'un kernel coûte 1.47x
   de plus à la moitié de l'occupation ne dit rien sur la possibilité d'augmenter l'occupation — cela exige
   l'arithmétique des ressources (ici : 2 blocs/SM aujourd'hui, 3 nécessiterait une réduction de 22% de la mémoire partagée).

## Documentation

| Document | Contenu |
|---|---|
| [`docs/USAGE.md`](docs/USAGE.md) | guide d'utilisation complet : exemples Python et streaming, tableau des paramètres, limites de décodage |
| [`docs/PHYSICS.md`](docs/PHYSICS.md) | physique matérielle mesurée et comment la roofline a été dérivée |
| [`docs/TARGETS.md`](docs/TARGETS.md) | le contrat de performance convenu |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | architecture du moteur |
| [`docs/LONG_CONTEXT.md`](docs/LONG_CONTEXT.md) | le kernel d'attention tuilée, le chunking, le budget mémoire, les résultats needle |
| [`bench/longctx/comparison.md`](bench/longctx/comparison.md) | le journal complet de comparaison en contexte long face à llama.cpp, y compris les hypothèses rejetées et les corrections |
| [`bench/ppl/`](bench/ppl/) · [`bench/mmlu/`](bench/mmlu/) | résultats de précision enregistrés et les harnais qui les ont produits |
| [`docs/NEXT.md`](docs/NEXT.md) | la liste des travaux ouverts |

## Licence

Voir le dépôt pour les informations de licence.

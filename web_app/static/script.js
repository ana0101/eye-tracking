function showLoading() {
    document.getElementById("loading").style.display = "block";
}

function updateWordColorAndTooltip(span, trt) {
    const norm = (trt - MIN_TRT) / (MAX_TRT - MIN_TRT + 1e-5);
    const hue = 120 - norm * 120; // 0 = red, 120 = green

    span.style.backgroundColor = `hsl(${hue}, 80%, 85%)`;
    span.dataset.trt = trt.toFixed(2);

    const tooltip = span.querySelector(".tooltip-text");
    tooltip.textContent = `TRT: ${trt.toFixed(2)} ms`;
}

document.addEventListener("DOMContentLoaded", () => {
    const simplifyWords = document.querySelectorAll(".simplify-word");

    // Per-word simplification
    simplifyWords.forEach(span => {
        span.addEventListener("click", async () => {
            const idx = +span.dataset.idx;
            const sentence = span.dataset.sentence;
            const wordTextEl = span.querySelector(".word-text");

            // Show inline loader
            span.classList.add("loading");

            try {
                const res = await fetch("/simplify", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ sentence, idx })
                });
                const { replacement, new_trt } = await res.json();

                wordTextEl.textContent = replacement;
                updateWordColorAndTooltip(span, new_trt);
            } catch (e) {
                console.error("Simplify error", e);
            } finally {
                // Hide inline loader
                span.classList.remove("loading");
            }
        });
    });
});

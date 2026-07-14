/**
 * OptiCrop — Client-side JavaScript
 *
 * Handles:
 *   - Quick-fill sample data
 *   - Client-side form validation with visual feedback
 *   - Loading overlay on submit
 *   - Navbar scroll effect
 *   - Intersection-observer scroll animations
 */

document.addEventListener("DOMContentLoaded", () => {

    // ── Quick-fill buttons ────────────────────────────────────────────────
    const qfButtons = document.querySelectorAll(".btn-quickfill[data-n]");
    const fields = {
        n:    document.getElementById("inputN"),
        p:    document.getElementById("inputP"),
        k:    document.getElementById("inputK"),
        temp: document.getElementById("inputTemp"),
        hum:  document.getElementById("inputHum"),
        ph:   document.getElementById("inputPh"),
        rain: document.getElementById("inputRain"),
    };

    qfButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            if (fields.n)    fields.n.value    = btn.dataset.n;
            if (fields.p)    fields.p.value    = btn.dataset.p;
            if (fields.k)    fields.k.value    = btn.dataset.k;
            if (fields.temp) fields.temp.value = btn.dataset.temp;
            if (fields.hum)  fields.hum.value  = btn.dataset.hum;
            if (fields.ph)   fields.ph.value   = btn.dataset.ph;
            if (fields.rain) fields.rain.value = btn.dataset.rain;

            // Trigger input event for floating-label activation
            Object.values(fields).forEach(f => {
                if (f) f.dispatchEvent(new Event("input", { bubbles: true }));
            });

            // Flash the active button
            qfButtons.forEach(b => b.style.borderColor = "");
            btn.style.borderColor = "var(--accent)";
        });
    });

    // ── Client-side validation ────────────────────────────────────────────
    const form = document.getElementById("predictForm");
    if (form) {
        form.addEventListener("submit", (e) => {
            let valid = true;
            form.querySelectorAll("input[required]").forEach(input => {
                if (!input.value.trim()) {
                    input.classList.add("is-invalid");
                    valid = false;
                } else {
                    input.classList.remove("is-invalid");
                }
            });

            if (!valid) {
                e.preventDefault();
                return;
            }

            // Show loading overlay
            const overlay = document.getElementById("loadingOverlay");
            if (overlay) overlay.classList.add("show");
        });

        // Remove invalid class on input
        form.querySelectorAll("input").forEach(input => {
            input.addEventListener("input", () => {
                input.classList.remove("is-invalid");
            });
        });
    }

    // ── Navbar scroll effect ──────────────────────────────────────────────
    const navbar = document.getElementById("main-navbar");
    if (navbar) {
        const onScroll = () => {
            navbar.classList.toggle("scrolled", window.scrollY > 40);
        };
        window.addEventListener("scroll", onScroll, { passive: true });
        onScroll();
    }

    // ── Intersection Observer — scroll animations ─────────────────────────
    const animElements = document.querySelectorAll(".animate-on-scroll");
    if (animElements.length && "IntersectionObserver" in window) {
        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add("visible");
                        observer.unobserve(entry.target);
                    }
                });
            },
            { threshold: 0.15 }
        );
        animElements.forEach(el => observer.observe(el));
    } else {
        // Fallback: show all immediately
        animElements.forEach(el => el.classList.add("visible"));
    }

    // ── Active nav link ───────────────────────────────────────────────────
    const path = window.location.pathname;
    document.querySelectorAll(".nav-link").forEach(link => {
        const href = link.getAttribute("href");
        if (href === path || (path === "/" && href === "/")) {
            link.classList.add("active");
        }
    });

});

/* ==========================================================================
   BrainCodes — project page
   1. Reveal on scroll
   2. Copy-to-clipboard buttons
   3. Hero background — discrete token lattice
   ========================================================================== */

(function () {
  "use strict";

  var REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* 1. Reveal on scroll
     ====================================================================== */

  function initReveal() {
    var targets = document.querySelectorAll(".reveal");

    if (REDUCED_MOTION || !("IntersectionObserver" in window)) {
      targets.forEach(function (el) { el.classList.add("is-in"); });
      return;
    }

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-in");
        observer.unobserve(entry.target);
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.08 });

    targets.forEach(function (el) { observer.observe(el); });
  }

  /* 2. Copy-to-clipboard buttons
     ====================================================================== */

  function copyWithSelection(text) {
    var previous = document.activeElement;
    var field = document.createElement("textarea");
    field.value = text;
    field.setAttribute("readonly", "");
    field.style.position = "fixed";
    field.style.opacity = "0";
    document.body.appendChild(field);
    field.select();

    var ok = true;
    try {
      document.execCommand("copy");
    } catch {
      ok = false;
    }

    document.body.removeChild(field);
    if (previous && previous.focus) previous.focus();
    return ok;
  }

  function flashCopied(button, name) {
    var label = button.textContent;
    button.textContent = "Copied";
    button.classList.add("is-done");
    var status = document.getElementById("copy-status");
    if (status) status.textContent = name + " copied to clipboard";
    setTimeout(function () {
      button.textContent = label;
      button.classList.remove("is-done");
      if (status) status.textContent = "";
    }, 1600);
  }

  function initCopyButtons() {
    document.querySelectorAll(".copy").forEach(function (button) {
      button.addEventListener("click", function () {
        var source = document.querySelector(button.getAttribute("data-copy"));
        if (!source) return;

        var block = source.closest("pre");
        var name = (block && block.getAttribute("aria-label")) || "Code";
        var text = source.innerText;
        var done = function () {
          flashCopied(button, name);
        };
        var fallback = function () {
          if (copyWithSelection(text)) done();
        };

        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(done).catch(fallback);
        } else {
          fallback();
        }
      });
    });
  }

  /* 3. Hero background — discrete token lattice
     ====================================================================== */

  var PALETTE = [
    [43, 92, 255],   /* blue */
    [0, 201, 214],   /* turquoise */
    [145, 71, 255]   /* lilac */
  ];

  var CODE_LEVELS = 5;

  function mix(a, b, t) {
    return [
      a[0] + (b[0] - a[0]) * t,
      a[1] + (b[1] - a[1]) * t,
      a[2] + (b[2] - a[2]) * t
    ];
  }

  function ramp(t) {
    var clamped = Math.min(Math.max(t, 0), 1);
    var pos = clamped * (PALETTE.length - 1);
    var i = Math.min(Math.floor(pos), PALETTE.length - 2);
    return mix(PALETTE[i], PALETTE[i + 1], pos - i);
  }

  function rgba(color, alpha) {
    return "rgba(" + (color[0] | 0) + "," + (color[1] | 0) + "," + (color[2] | 0) + "," +
      alpha.toFixed(3) + ")";
  }

  function roundedRect(ctx, cx, cy, size) {
    var r = Math.min(2.4, size / 2);
    var x = cx - size / 2;
    var y = cy - size / 2;

    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + size, y, x + size, y + size, r);
    ctx.arcTo(x + size, y + size, x, y + size, r);
    ctx.arcTo(x, y + size, x, y, r);
    ctx.arcTo(x, y, x + size, y, r);
    ctx.closePath();
  }

  function initLattice() {
    var canvas = document.getElementById("token-canvas");
    if (!canvas || REDUCED_MOTION) return;

    var ctx = canvas.getContext("2d");
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var width = 0;
    var height = 0;
    var startedAt = performance.now();
    var running = true;

    function resize() {
      var rect = canvas.getBoundingClientRect();
      width = Math.max(rect.width, 1);
      height = Math.max(rect.height, 1);
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function draw(now) {
      if (!running) return;

      var t = (now - startedAt) / 1000;
      ctx.clearRect(0, 0, width, height);

      /* lattice geometry: rows = channels, columns = code frames */
      var cell = width > 1100 ? 34 : (width > 700 ? 30 : 26);
      var cols = Math.ceil(width / cell) + 1;
      var rows = Math.max(6, Math.min(18, Math.floor(height / cell)));
      var offsetY = (height - (rows - 1) * cell) / 2;
      var drift = (t * 9) % cell;

      for (var r = 0; r < rows; r++) {
        var y = offsetY + r * cell;
        var rowPhase = r * 0.62;
        var rowFade = 1 - Math.abs((r / (rows - 1)) - 0.5) * 1.15;
        var prevX = null;
        var prevActivation = 0;

        for (var c = 0; c < cols; c++) {
          var x = c * cell - drift;
          var u = x / Math.max(width, 1);

          /* overlapping activation fields, then quantized to discrete levels */
          var field =
            Math.sin(u * 6.1 - t * 0.85 + rowPhase) * 0.55 +
            Math.sin(u * 2.3 + t * 0.42 - rowPhase * 0.7) * 0.35 +
            Math.sin((u * 11.7 + r * 0.9) - t * 1.35) * 0.22;

          var level = Math.round(((field + 1.12) / 2.24) * (CODE_LEVELS - 1)) / (CODE_LEVELS - 1);
          level = Math.min(Math.max(level, 0), 1);

          var activation = level * Math.max(rowFade, 0.12);
          var color = ramp(0.12 + level * 0.76 + r / rows * 0.14);

          /* connector between consecutive active tokens */
          if (prevX !== null && activation > 0.45 && prevActivation > 0.45) {
            ctx.beginPath();
            ctx.moveTo(prevX + 4, y);
            ctx.lineTo(x - 4, y);
            ctx.strokeStyle = rgba(color, 0.10 * activation);
            ctx.lineWidth = 1;
            ctx.stroke();
          }

          var size = 2.2 + activation * 6.4;
          roundedRect(ctx, x, y, size);
          ctx.fillStyle = rgba(color, 0.06 + activation * 0.42);
          ctx.fill();

          /* occasional "selected code" halo */
          if (level > 0.99 && ((r * 31 + c * 17) % 23 === (Math.floor(t * 1.6) % 23))) {
            ctx.beginPath();
            ctx.arc(x, y, size * 1.55, 0, Math.PI * 2);
            ctx.strokeStyle = rgba(color, 0.30 * rowFade);
            ctx.lineWidth = 1;
            ctx.stroke();
          }

          prevX = x;
          prevActivation = activation;
        }
      }

      requestAnimationFrame(draw);
    }

    if ("ResizeObserver" in window) {
      new ResizeObserver(resize).observe(canvas);
    } else {
      window.addEventListener("resize", resize);
    }

    document.addEventListener("visibilitychange", function () {
      if (document.hidden) {
        running = false;
      } else if (!running) {
        running = true;
        startedAt = performance.now() - 1000;
        requestAnimationFrame(draw);
      }
    });

    resize();
    requestAnimationFrame(draw);
  }

  initReveal();
  initCopyButtons();
  initLattice();
})();

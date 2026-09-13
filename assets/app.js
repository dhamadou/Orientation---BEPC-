(function () {
  "use strict";

  var PAGE_SIZE = 30;
  var state = {
    all: [],
    filtered: [],
    shown: 0,
  };

  // --- lightweight fuzzy matching (no external dependency) ---
  function levenshtein(a, b) {
    if (a === b) return 0;
    var al = a.length, bl = b.length;
    if (al === 0) return bl;
    if (bl === 0) return al;
    var prev = new Array(bl + 1);
    for (var j = 0; j <= bl; j++) prev[j] = j;
    for (var i = 1; i <= al; i++) {
      var cur = [i];
      for (var k = 1; k <= bl; k++) {
        var cost = a.charAt(i - 1) === b.charAt(k - 1) ? 0 : 1;
        cur[k] = Math.min(prev[k] + 1, cur[k - 1] + 1, prev[k - 1] + cost);
      }
      prev = cur;
    }
    return prev[bl];
  }

  function tokenScore(qTok, nTok) {
    if (!qTok || !nTok) return 0;
    if (nTok.indexOf(qTok) !== -1 || qTok.indexOf(nTok) !== -1) return 1;
    var maxLen = Math.max(qTok.length, nTok.length);
    var allowed = qTok.length <= 3 ? 1 : Math.max(1, Math.floor(maxLen / 4));
    var d = levenshtein(qTok, nTok);
    if (d <= allowed) return 1 - d / (maxLen + 1);
    return 0;
  }

  function matchScore(queryTokens, nameTokens) {
    var total = 0;
    for (var i = 0; i < queryTokens.length; i++) {
      var best = 0;
      for (var j = 0; j < nameTokens.length; j++) {
        var s = tokenScore(queryTokens[i], nameTokens[j]);
        if (s > best) best = s;
      }
      if (best === 0) return 0; // every query token must match something
      total += best;
    }
    return total / queryTokens.length;
  }

  function fuzzySearch(list, query) {
    var qTokens = query.split(/\s+/).filter(Boolean);
    if (!qTokens.length) return list;
    var scored = [];
    for (var i = 0; i < list.length; i++) {
      var r = list[i];
      var score = matchScore(qTokens, r._nomTokens);
      if (score > 0) scored.push({ item: r, score: score });
    }
    scored.sort(function (a, b) { return b.score - a.score; });
    return scored.map(function (s) { return s.item; });
  }

  var els = {
    search: document.getElementById("search-input"),
    year: document.getElementById("year-select"),
    school: document.getElementById("school-select"),
    region: document.getElementById("region-select"),
    sexe: document.getElementById("sexe-select"),
    origin: document.getElementById("origin-select"),
    resultCount: document.getElementById("result-count"),
    resultsList: document.getElementById("results-list"),
    loadMoreWrap: document.getElementById("load-more-wrap"),
    loadMore: document.getElementById("load-more"),
    schoolBrowse: document.getElementById("school-browse-list"),
    toggleCoverage: document.getElementById("toggle-coverage"),
    coverageDetail: document.getElementById("coverage-detail"),
  };

  function normalize(s) {
    if (!s) return "";
    return s
      .toString()
      .normalize("NFD")
      .replace(/[̀-ͯ]/g, "")
      .toLowerCase()
      .trim();
  }

  function uniqueSorted(arr) {
    return Array.from(new Set(arr.filter(Boolean))).sort(function (a, b) {
      return a.localeCompare(b, "fr");
    });
  }

  function fillSelect(select, values, placeholder) {
    values.forEach(function (v) {
      var opt = document.createElement("option");
      opt.value = v;
      opt.textContent = v;
      select.appendChild(opt);
    });
  }

  function loadJSON(path) {
    return fetch(path).then(function (r) {
      if (!r.ok) throw new Error("Impossible de charger " + path);
      return r.json();
    });
  }

  function init() {
    Promise.all([
      loadJSON("data/2026-2027.json").catch(function () { return []; }),
      loadJSON("data/2025-2026.json").catch(function () { return []; }),
      loadJSON("data/meta.json").catch(function () { return null; }),
    ]).then(function (res) {
      var d1 = res[0].map(function (r) { return Object.assign({}, r, { annee: "2026-2027" }); });
      var d2 = res[1].map(function (r) { return Object.assign({}, r, { annee: "2025-2026" }); });
      state.all = d1.concat(d2).map(function (r, i) {
        r._id = i;
        r._nomNorm = normalize(r.nom_prenom);
        r._nomTokens = r._nomNorm.split(/\s+/).filter(Boolean);
        return r;
      });
      renderMeta(res[2]);
      setupFilters();
      setupBrowse();
      bindEvents();
      updateResults();
    }).catch(function (err) {
      els.resultsList.innerHTML =
        '<p style="color:#a3305a">Erreur de chargement des données : ' + err.message + "</p>";
    });
  }

  function renderMeta(meta) {
    if (!meta) return;
    var html = "";
    (meta.datasets || []).forEach(function (ds) {
      html += "<p><strong>" + ds.annee + "</strong> — " + ds.description + "</p><ul>";
      (ds.sections || []).forEach(function (s) {
        html += "<li>" + s + "</li>";
      });
      html += "</ul>";
    });
    els.coverageDetail.innerHTML = html || "<p>Détails non disponibles.</p>";
  }

  function setupFilters() {
    var years = uniqueSorted(state.all.map(function (r) { return r.annee; }));
    fillSelect(els.year, years);

    var schools = uniqueSorted(state.all.map(function (r) { return r.ecole_orientation; }));
    fillSelect(els.school, schools);

    var regions = uniqueSorted(state.all.map(function (r) { return r.region_origine; }));
    fillSelect(els.region, regions);

    var origins = uniqueSorted(state.all.map(function (r) { return r.etablissement_origine; }));
    fillSelect(els.origin, origins);
  }

  function setupBrowse() {
    var counts = {};
    state.all.forEach(function (r) {
      var key = r.ecole_orientation || "Inconnu";
      counts[key] = (counts[key] || 0) + 1;
    });
    var schools = Object.keys(counts).sort(function (a, b) { return a.localeCompare(b, "fr"); });
    els.schoolBrowse.innerHTML = "";
    schools.forEach(function (s) {
      var chip = document.createElement("div");
      chip.className = "school-chip";
      chip.innerHTML = s + ' <span class="count">(' + counts[s] + ")</span>";
      chip.addEventListener("click", function () {
        els.school.value = s;
        els.search.value = "";
        document.querySelector(".filters").open = true;
        updateResults();
        window.scrollTo({ top: els.search.offsetTop - 80, behavior: "smooth" });
      });
      els.schoolBrowse.appendChild(chip);
    });
  }

  function bindEvents() {
    els.search.addEventListener("input", debounce(updateResults, 150));
    [els.year, els.school, els.region, els.sexe, els.origin].forEach(function (el) {
      el.addEventListener("change", updateResults);
    });
    els.loadMore.addEventListener("click", function () {
      state.shown += PAGE_SIZE;
      renderList();
    });
    els.toggleCoverage.addEventListener("click", function () {
      els.coverageDetail.classList.toggle("hidden");
      els.toggleCoverage.textContent = els.coverageDetail.classList.contains("hidden")
        ? "Voir la couverture exacte des données ▾"
        : "Masquer les détails ▴";
    });
  }

  function debounce(fn, ms) {
    var t;
    return function () {
      clearTimeout(t);
      var args = arguments;
      t = setTimeout(function () { fn.apply(null, args); }, ms);
    };
  }

  function applyFilters(list) {
    var year = els.year.value;
    var school = els.school.value;
    var region = els.region.value;
    var sexe = els.sexe.value;
    var origin = els.origin.value;
    return list.filter(function (r) {
      if (year && r.annee !== year) return false;
      if (school && r.ecole_orientation !== school) return false;
      if (region && r.region_origine !== region) return false;
      if (sexe && r.sexe !== sexe) return false;
      if (origin && r.etablissement_origine !== origin) return false;
      return true;
    });
  }

  function hasActiveFilter() {
    return !!(els.year.value || els.school.value || els.region.value || els.sexe.value || els.origin.value);
  }

  function updateResults() {
    var q = normalize(els.search.value);
    var active = q.length >= 2 || hasActiveFilter();
    if (!active) {
      state.filtered = [];
      state.shown = 0;
      els.resultCount.textContent = "Tapez un nom ou choisissez un filtre pour commencer.";
      els.resultsList.innerHTML = "";
      els.loadMoreWrap.classList.add("hidden");
      return;
    }
    var base = q.length >= 2 ? fuzzySearch(state.all, q) : state.all;
    state.filtered = applyFilters(base);
    state.shown = PAGE_SIZE;
    renderList();
  }

  function renderList() {
    var total = state.filtered.length;
    var q = els.search.value.trim();
    if (total === 0) {
      els.resultCount.textContent = q
        ? "Aucun résultat pour « " + q + " ». Vérifiez l'orthographe ou essayez juste le prénom / nom de famille."
        : "Aucun résultat pour ces filtres.";
    } else {
      els.resultCount.textContent =
        total + " résultat" + (total > 1 ? "s" : "") + (q ? " pour « " + q + " »" : "");
    }

    var toShow = state.filtered.slice(0, state.shown);
    els.resultsList.innerHTML = toShow.map(renderCard).join("");
    els.loadMoreWrap.classList.toggle("hidden", state.shown >= total);
  }

  function esc(s) {
    if (!s) return "";
    var d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function renderCard(r) {
    var sexeLabel = r.sexe === "F" ? "Féminin" : r.sexe === "M" ? "Masculin" : "?";
    return (
      '<div class="result-card">' +
      '<p class="result-name">' + esc(r.nom_prenom || "(nom illisible)") + "</p>" +
      '<div class="result-badges">' +
      '<span class="badge year">' + esc(r.annee) + "</span>" +
      '<span class="badge sexe-' + esc(r.sexe) + '">' + sexeLabel + "</span>" +
      "</div>" +
      '<div class="result-grid">' +
      '<div><div class="label">Orienté(e) vers</div><div class="value school">' + esc(r.ecole_orientation) + "</div></div>" +
      '<div><div class="label">Établissement d\'origine</div><div class="value">' + esc(r.etablissement_origine) + "</div></div>" +
      '<div><div class="label">Région d\'origine</div><div class="value">' + esc(r.region_origine) + "</div></div>" +
      '<div><div class="label">Date et lieu de naissance</div><div class="value">' +
      esc(r.date_naissance) + (r.lieu_naissance ? " à " + esc(r.lieu_naissance) : "") + "</div></div>" +
      "</div>" +
      "</div>"
    );
  }

  init();
})();

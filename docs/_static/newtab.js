// Open links from sphinx-design cards (`:link:` option) in a new tab.
// The card renders its click target as an <a class="sd-stretched-link">.
document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("a.sd-stretched-link").forEach(function (link) {
        if (link.href && /^https?:\/\//.test(link.href)) {
            link.target = "_blank";
            link.rel = "noopener noreferrer";
        }
    });
});

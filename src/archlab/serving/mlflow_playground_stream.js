(function () {
  if (window.__archlabPlaygroundStream) {
    return;
  }
  window.__archlabPlaygroundStream = true;

  var orig = window.fetch.bind(window);
  var hideTimer = null;

  function isChatCompletionsUrl(url) {
    if (!url) {
      return false;
    }
    var path = String(url).split("?")[0];
    if (path.indexOf("://") >= 0) {
      try {
        path = new URL(path, window.location.href).pathname;
      } catch (err) {
        /* keep path */
      }
    }
    return /(^|\/)gateway\/mlflow\/v1\/chat\/completions\/?$/.test(path);
  }

  function requestUrl(input) {
    if (typeof input === "string") {
      return input;
    }
    if (input && typeof input.url === "string") {
      return input.url;
    }
    return "";
  }

  function mountLiveCard() {
    var el = document.getElementById("archlab-playground-stream");
    if (el) {
      return el;
    }
    el = document.createElement("div");
    el.id = "archlab-playground-stream";
    el.setAttribute("data-archlab-stream", "1");
    el.style.cssText = [
      "margin:12px 0 16px",
      "padding:16px",
      "border:1px solid rgba(15,23,42,.12)",
      "border-radius:8px",
      "background:#f8fafc",
      "color:#0f172a",
      "font:13px/1.5 ui-sans-serif, system-ui, sans-serif",
      "white-space:pre-wrap",
      "word-break:break-word",
      "max-height:48vh",
      "overflow:auto",
    ].join(";");
    var label = document.createElement("div");
    label.textContent = "Assistant · streaming";
    label.style.cssText = "color:#64748b;font-size:12px;margin-bottom:8px;";
    var body = document.createElement("div");
    body.id = "archlab-playground-stream-body";
    el.appendChild(label);
    el.appendChild(body);
    var submit = Array.prototype.find.call(document.querySelectorAll("button"), function (button) {
      return ((button.textContent || "").trim() === "Submit");
    });
    var row = submit && submit.parentElement;
    var host = row && row.parentElement;
    if (host && row) {
      host.insertBefore(el, row);
    } else {
      (document.querySelector("#root") || document.body).appendChild(el);
    }
    return el;
  }

  function showLive(text) {
    if (hideTimer) {
      window.clearTimeout(hideTimer);
      hideTimer = null;
    }
    mountLiveCard();
    var body = document.getElementById("archlab-playground-stream-body");
    if (body) {
      body.textContent = text || "Streaming…";
    }
  }

  function hideLive() {
    var el = document.getElementById("archlab-playground-stream");
    if (el && el.parentNode) {
      el.parentNode.removeChild(el);
    }
  }

  function hideLiveSoon() {
    if (hideTimer) {
      window.clearTimeout(hideTimer);
    }
    hideTimer = window.setTimeout(hideLive, 250);
  }

  function mergeToolCalls(acc, deltas) {
    if (!Array.isArray(deltas)) {
      return;
    }
    for (var i = 0; i < deltas.length; i += 1) {
      var delta = deltas[i] || {};
      var index = typeof delta.index === "number" ? delta.index : acc.length;
      while (acc.length <= index) {
        acc.push({ type: "function", function: { name: "", arguments: "" } });
      }
      var dest = acc[index];
      if (delta.id) {
        dest.id = delta.id;
      }
      if (delta.type) {
        dest.type = delta.type;
      }
      dest.function = dest.function || { name: "", arguments: "" };
      if (delta.function && delta.function.name) {
        dest.function.name += delta.function.name;
      }
      if (delta.function && delta.function.arguments) {
        dest.function.arguments += delta.function.arguments;
      }
    }
  }

  async function readBody(input, init) {
    if (init && typeof init.body === "string") {
      return init.body;
    }
    if (input && typeof input.clone === "function") {
      return await input.clone().text();
    }
    return null;
  }

  async function consumeSse(response, onText) {
    var reader = response.body.getReader();
    var decoder = new TextDecoder();
    var buffer = "";
    var acc = {
      content: "",
      tool_calls: [],
      role: "assistant",
      id: null,
      model: null,
      created: null,
      usage: null,
      finish_reason: null,
    };
    while (true) {
      var chunk = await reader.read();
      if (chunk.done) {
        break;
      }
      buffer += decoder.decode(chunk.value, { stream: true });
      var sep;
      while ((sep = buffer.indexOf("\n")) >= 0) {
        var line = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 1);
        if (line.charAt(line.length - 1) === "\r") {
          line = line.slice(0, -1);
        }
        if (line.indexOf("data:") !== 0) {
          continue;
        }
        var payload = line.slice(5).trim();
        if (!payload || payload === "[DONE]") {
          continue;
        }
        var event;
        try {
          event = JSON.parse(payload);
        } catch (err) {
          continue;
        }
        if (event.error) {
          throw new Error((event.error && event.error.message) || "gateway stream error");
        }
        if (event.id) {
          acc.id = event.id;
        }
        if (event.model) {
          acc.model = event.model;
        }
        if (event.created) {
          acc.created = event.created;
        }
        if (event.usage) {
          acc.usage = event.usage;
        }
        var choice = event.choices && event.choices[0];
        if (!choice) {
          continue;
        }
        if (choice.finish_reason) {
          acc.finish_reason = choice.finish_reason;
        }
        var delta = choice.delta || {};
        if (delta.role) {
          acc.role = delta.role;
        }
        if (delta.content) {
          acc.content += delta.content;
          onText(acc.content);
        }
        if (delta.tool_calls) {
          mergeToolCalls(acc.tool_calls, delta.tool_calls);
          onText(acc.content || ("Calling " + acc.tool_calls.map(function (call) {
            return (call.function && call.function.name) || "tool";
          }).join(", ")));
        }
      }
    }
    return acc;
  }

  function aggregatedFrom(acc, fallbackModel) {
    var message = {
      role: acc.role || "assistant",
      content: acc.content || null,
    };
    if (acc.tool_calls.length) {
      message.tool_calls = acc.tool_calls;
    }
    return {
      id: acc.id,
      object: "chat.completion",
      created: acc.created || Math.floor(Date.now() / 1000),
      model: acc.model || fallbackModel,
      choices: [
        {
          index: 0,
          message: message,
          finish_reason: acc.finish_reason || "stop",
        },
      ],
      usage: acc.usage,
    };
  }

  window.__archlabPlaygroundChat = async function (request, fetchOrFail, getAjaxUrl) {
    var payload = Object.assign({}, request || {}, { stream: true });
    var send = fetchOrFail || orig;
    var url = getAjaxUrl ? getAjaxUrl("gateway/mlflow/v1/chat/completions") : "gateway/mlflow/v1/chat/completions";
    showLive("Streaming…");
    try {
      var response = await send(url, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify(payload),
      });
      var type = ((response.headers && response.headers.get("content-type")) || "").toLowerCase();
      if (!response.ok || !response.body || type.indexOf("event-stream") < 0) {
        return response.json ? await response.json() : response;
      }
      var acc = await consumeSse(response, showLive);
      return aggregatedFrom(acc, payload.model);
    } finally {
      hideLiveSoon();
    }
  };

  window.fetch = async function (input, init) {
    var url = requestUrl(input);
    if (!isChatCompletionsUrl(url)) {
      return orig(input, init);
    }
    var raw;
    try {
      raw = await readBody(input, init);
    } catch (err) {
      return orig(input, init);
    }
    var body;
    try {
      body = JSON.parse(raw);
    } catch (err) {
      return orig(input, init);
    }
    if (!body || typeof body !== "object" || body.stream === true) {
      return orig(input, init);
    }
    body.stream = true;
    var headers = new Headers((init && init.headers) || (input && input.headers) || undefined);
    headers.set("Content-Type", "application/json");
    headers.set("Accept", "text/event-stream");
    showLive("Streaming…");
    try {
      var response = await orig(url, {
        method: "POST",
        headers: headers,
        body: JSON.stringify(body),
        credentials: (init && init.credentials) || "same-origin",
        signal: (init && init.signal) || (input && input.signal) || undefined,
      });
      var type = (response.headers.get("content-type") || "").toLowerCase();
      if (!response.ok || !response.body || type.indexOf("event-stream") < 0) {
        return response;
      }
      var acc = await consumeSse(response, showLive);
      return new Response(JSON.stringify(aggregatedFrom(acc, body.model)), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    } finally {
      hideLiveSoon();
    }
  };
})();

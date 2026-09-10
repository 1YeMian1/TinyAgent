/* ==============================================
   TinyLLM-Story — API 接口层
   ============================================== */

const API = {
    // 同源部署时无需配置；分离部署可在加载本脚本前设置 window.TINYLLM_API_BASE_URL。
    BASE_URL: window.TINYLLM_API_BASE_URL || `${window.location.origin}/api/v1`,

    async request(endpoint, options = {}) {
        const url = `${this.BASE_URL}${endpoint}`;
        const headers = {
            'Content-Type': 'application/json',
            ...options.headers,
        };
        const token = Auth.getToken();
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }

        const response = await fetch(url, {
            ...options,
            headers,
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({ message: '请求失败' }));
            if (response.status === 401) {
                Auth.logout();
            }
            throw new Error(err.message || err.detail || `HTTP ${response.status}`);
        }

        return response.status === 204 ? null : response.json();
    },

    // --- Categories ---
    categories: {
        list() {
            return API.request('/categories');
        },
    },

    // --- Auth ---
    auth: {
        login(email, password) {
            return API.request('/auth/login', {
                method: 'POST',
                body: JSON.stringify({ email, password }),
            });
        },
        register(username, email, password) {
            return API.request('/auth/register', {
                method: 'POST',
                body: JSON.stringify({ username, email, password }),
            });
        },
        profile() {
            return API.request('/auth/profile');
        },
    },

    // --- Stories ---
    stories: {
        list(params = {}) {
            const query = new URLSearchParams(Object.entries(params).filter(([, value]) => value !== undefined && value !== null && value !== '')).toString();
            return API.request(`/stories${query ? '?' + query : ''}`);
        },
        get(id) {
            return API.request(`/stories/${id}`);
        },
        favorites(params = {}) {
            const query = new URLSearchParams(Object.entries(params).filter(([, value]) => value !== undefined && value !== null && value !== '')).toString();
            return API.request(`/stories/favorites${query ? '?' + query : ''}`);
        },
        toggleFavorite(id) {
            return API.request(`/stories/${id}/favorite`, { method: 'POST' });
        },
        history(params = {}) {
            const query = new URLSearchParams(Object.entries(params).filter(([, value]) => value !== undefined && value !== null && value !== '')).toString();
            return API.request(`/stories/history${query ? '?' + query : ''}`);
        },
    },

    // --- Generate ---
    generate: {
        create(data, options = {}) {
            return API.request('/generate/story', {
                method: 'POST',
                body: JSON.stringify(data),
                ...options,
            });
        },

        /**
         * SSE 流式故事生成 — 优先使用 EventSource（原生流式无缓冲），
         * 回退到 fetch + ReadableStream（支持 AbortController 和 auth header）。
         *
         * @param {Object}   data    - 生成参数
         * @param {Function} onChunk - (text: string) => void
         * @param {Function} onDone  - (result: {full_text, category}) => void
         * @param {Function} onError - (error: Error | string) => void
         * @param {AbortSignal} [signal] - 取消信号（仅 fetch 模式使用）
         * @returns {{ close: Function }} 返回可调用 close() 的对象
         */
        streamFetch(data, onChunk, onDone, onError, signal) {
            const params = new URLSearchParams();
            for (const [k, v] of Object.entries(data)) {
                if (v !== undefined && v !== null && v !== '') params.append(k, v);
            }

            // 将 auth token 作为 query param 传递（EventSource 不支持自定义 header）
            const token = Auth.getToken();
            if (token) params.append('token', token);

            const url = `${API.BASE_URL}/generate/story/stream?${params.toString()}`;
            const es = new EventSource(url);

            es.onmessage = (event) => {
                try {
                    const chunk = JSON.parse(event.data);
                    if (chunk.error) {
                        es.close();
                        onError && onError(chunk.error);
                        return;
                    }
                    if (chunk.done) {
                        es.close();
                        onDone && onDone(chunk);
                        return;
                    }
                    if (chunk.text) {
                        onChunk && onChunk(chunk.text);
                    }
                    // ready 事件静默处理
                } catch (_) { /* ignore parse errors */ }
            };

            es.onerror = () => {
                // EventSource 在流正常结束时也可能触发 onerror（服务器关闭连接）
                // 此时 readyState 已经是 CLOSED，正常结束不需要报错
                if (es.readyState === EventSource.CLOSED) {
                    // 如果还没收到 done，发送一个默认完成事件
                    if (!es._doneReceived) {
                        es._doneReceived = true;
                        es.close();
                        onDone && onDone({ done: true, full_text: '', category: data.category || '' });
                    }
                } else {
                    es.close();
                    onError && onError(new Error('SSE 连接中断'));
                }
            };

            // 支持外部 AbortController 取消
            if (signal) {
                const abortHandler = () => {
                    es.close();
                    onError && onError(new DOMException('用户取消', 'AbortError'));
                };
                if (signal.aborted) {
                    abortHandler();
                    return { close() { es.close(); } };
                }
                signal.addEventListener('abort', abortHandler, { once: true });
            }

            return { close() { es.close(); } };
        },

        stream(data, onChunk, onDone, onError) {
            // 旧版兼容：内部委托给 streamFetch
            return API.generate.streamFetch(data, onChunk, onDone, onError);
        },

        images(storyId, count = 3) {
            return API.request('/generate/images', {
                method: 'POST',
                body: JSON.stringify({ story_id: storyId, count }),
            });
        },
        voice(storyId, voiceId) {
            return API.request('/generate/voice', {
                method: 'POST',
                body: JSON.stringify({ story_id: storyId, voice_id: voiceId }),
            });
        },
    },

    // --- Voices ---
    voices: {
        list() {
            return API.request('/voices');
        },
    },

    // --- Assistant（智能助手：RAG 问答 + Agent 工具） ---
    assistant: {
        createSession(title = '新会话') {
            return API.request('/assistant/sessions', {
                method: 'POST',
                body: JSON.stringify({ title }),
            });
        },
        listSessions() {
            return API.request('/assistant/sessions');
        },
        getSession(id) {
            return API.request(`/assistant/sessions/${id}`);
        },
        deleteSession(id) {
            return API.request(`/assistant/sessions/${id}`, { method: 'DELETE' });
        },
        chat(id, message) {
            return API.request(`/assistant/sessions/${id}/chat`, {
                method: 'POST',
                body: JSON.stringify({ message }),
            });
        },
    },
};

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
        stream(data, onChunk, onDone, onError) {
            // SSE 流式生成
            const params = new URLSearchParams(data).toString();
            const url = `${API.BASE_URL}/generate/story/stream?${params}`;
            const eventSource = new EventSource(url);
            // Note: EventSource doesn't support custom headers natively, so auth token is not sent.
            // For production, pass token as query param or use fetch + ReadableStream.

            eventSource.onmessage = (event) => {
                const chunk = JSON.parse(event.data);
                if (chunk.done) {
                    eventSource.close();
                    onDone && onDone(chunk);
                } else {
                    onChunk && onChunk(chunk.text);
                }
            };
            eventSource.onerror = (err) => {
                eventSource.close();
                onError && onError(err);
            };

            return eventSource;
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

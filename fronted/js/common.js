/* ==============================================
   TinyLLM-Story — 公共 JavaScript 工具
   ============================================== */

// --- Storage ---
const Storage = {
    get(key) {
        try { return JSON.parse(localStorage.getItem(key)); }
        catch { return null; }
    },
    set(key, val) {
        localStorage.setItem(key, JSON.stringify(val));
    },
    remove(key) {
        localStorage.removeItem(key);
    },
    clear() {
        localStorage.clear();
    }
};

// --- Auth Helpers ---
const Auth = {
    KEY: 'tinyllm_user',

    isLoggedIn() {
        return !!Storage.get(this.KEY);
    },

    getUser() {
        return Storage.get(this.KEY);
    },

    login(authResponse) {
        const user = authResponse && authResponse.user
            ? { ...authResponse.user, token: authResponse.token }
            : authResponse;
        Storage.set(this.KEY, user);
    },

    logout() {
        Storage.remove(this.KEY);
    },

    getToken() {
        const user = this.getUser();
        return user ? user.token : null;
    },

    requireAuth() {
        if (!this.isLoggedIn()) {
            window.location.href = 'login.html';
            return false;
        }
        return true;
    }
};

// --- Toast Notifications ---
const Toast = {
    container: null,

    _ensureContainer() {
        if (!this.container) {
            this.container = document.createElement('div');
            this.container.className = 'toast-container';
            document.body.appendChild(this.container);
        }
    },

    show(message, type = 'info', duration = 3000) {
        this._ensureContainer();
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        const icons = { success: '✅', error: '❌', info: '💡' };
        toast.textContent = `${icons[type] || ''} ${message}`;
        this.container.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(100px)';
            toast.style.transition = '0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, duration);
    },

    success(msg) { this.show(msg, 'success'); },
    error(msg) { this.show(msg, 'error', 5000); },
    info(msg) { this.show(msg, 'info'); }
};

// --- Debounce ---
function debounce(fn, delay = 300) {
    let timer;
    return function (...args) {
        clearTimeout(timer);
        timer = setTimeout(() => fn.apply(this, args), delay);
    };
}

// --- Formatting ---
function formatTime(seconds) {
    if (!seconds || isNaN(seconds)) return '00:00';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function formatDate(dateStr) {
    const d = new Date(dateStr);
    if (isNaN(d.getTime())) return '未知日期';
    const now = new Date();
    const diff = now - d;
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);
    if (minutes < 1) return '刚刚';
    if (minutes < 60) return `${minutes}分钟前`;
    if (hours < 24) return `${hours}小时前`;
    if (days < 7) return `${days}天前`;
    return d.toLocaleDateString('zh-CN');
}

function truncateText(text, maxLen = 100) {
    if (!text) return '';
    return text.length > maxLen ? text.slice(0, maxLen) + '...' : text;
}

function estimateReadTime(text) {
    if (!text) return 0;
    return Math.max(1, Math.round(text.length / 300));
}

function getStoryLengthLabel(chars) {
    if (!chars) return '短篇';
    if (chars <= 200) return '短篇';
    if (chars <= 500) return '中篇';
    if (chars <= 1000) return '长篇';
    return '超长篇';
}

// --- HTML Escaping ---
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// --- Query String ---
function getQueryParam(name) {
    const params = new URLSearchParams(window.location.search);
    return params.get(name);
}

// --- Update Navbar ---
function updateNavbar() {
    const navUserDiv = document.getElementById('nav-user-content');
    if (!navUserDiv) return;

    if (Auth.isLoggedIn()) {
        const user = Auth.getUser();
        navUserDiv.innerHTML = `
            <a href="profile.html" class="avatar" title="${escapeHtml(user.username || '用户')}">
                ${(user.username || '用')[0]}
            </a>
            <span style="font-size:14px;font-weight:500;">${escapeHtml(user.username || '用户')}</span>
            <button class="btn-outline" onclick="Auth.logout();window.location.reload();" style="padding:5px 16px;font-size:13px;">退出</button>
        `;
    } else {
        navUserDiv.innerHTML = `
            <a href="login.html" class="btn-outline">登录</a>
        `;
    }
}

// --- Set Active Nav Link ---
function setActiveNav() {
    const path = window.location.pathname;
    const pageName = path.split('/').pop() || 'index.html';
    document.querySelectorAll('.nav-links a').forEach(link => {
        const href = link.getAttribute('href');
        if (href && (path.endsWith(href) || (pageName === 'index.html' && href === 'index.html'))) {
            link.classList.add('active');
        }
    });
}

// --- Init on DOM Ready ---
document.addEventListener('DOMContentLoaded', () => {
    updateNavbar();
    setActiveNav();
});

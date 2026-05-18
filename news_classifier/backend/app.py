
import os, re, math, pickle
import numpy as np
from collections import defaultdict, Counter
import streamlit as st

# ══════════════════════════════════════════════════════════════
#  ALL ML CLASSES — must match train_model.py exactly
#  (pickle needs these defined here to load the .pkl file)
# ══════════════════════════════════════════════════════════════

class TextPreprocessor:
    STOPWORDS = set([
        "a","an","the","and","or","but","in","on","at","to","for","of","with",
        "by","from","is","are","was","were","be","been","being","have","has",
        "had","do","does","did","will","would","could","should","may","might",
        "shall","can","this","that","these","those","it","its","he","she","they",
        "we","you","i","me","him","her","us","them","my","your","his","their",
        "our","as","if","so","not","no","nor","up","out","about","into","after",
        "before","between","during","through","also","than","then","when","where",
        "which","who","whom","what","how","all","each","every","both","few","more",
        "most","other","some","such","over","under","again","further","too","very",
        "just","new","said","says","say",
    ])
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"\d+", " NUM ", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        tokens = [t for t in text.split() if t not in self.STOPWORDS and len(t) > 2]
        return " ".join([self._stem(t) for t in tokens])
    def _stem(self, word):
        for sfx in ["ing","tion","ness","ment","ity","ies","ers","est","ed","ly","es","s"]:
            if word.endswith(sfx) and len(word)-len(sfx) >= 3:
                return word[:-len(sfx)]
        return word


class TFIDFVectorizer:
    def __init__(self, max_features=3000, ngram_range=(1,2)):
        self.max_features = max_features
        self.ngram_range  = ngram_range
        self.vocabulary_  = {}
        self.idf_         = {}
    def _ngrams(self, tokens):
        ng = list(tokens)
        if self.ngram_range[1] >= 2:
            ng += [tokens[i]+"_"+tokens[i+1] for i in range(len(tokens)-1)]
        return ng
    def fit(self, corpus):
        tokenized = [doc.split() for doc in corpus]
        ngrammed  = [self._ngrams(t) for t in tokenized]
        df = defaultdict(int)
        for doc in ngrammed:
            for term in set(doc): df[term] += 1
        top = sorted(df.items(), key=lambda x: -x[1])[:self.max_features]
        self.vocabulary_ = {t: i for i,(t,_) in enumerate(top)}
        N = len(corpus)
        self.idf_ = {t: math.log((N+1)/(f+1))+1 for t,f in df.items() if t in self.vocabulary_}
        return self
    def transform(self, corpus):
        M = np.zeros((len(corpus), len(self.vocabulary_)))
        for i, doc in enumerate(corpus):
            tokens = self._ngrams(doc.split())
            tf = Counter(tokens)
            total = sum(tf.values()) or 1
            for term, cnt in tf.items():
                if term in self.vocabulary_:
                    M[i, self.vocabulary_[term]] = (cnt/total) * self.idf_.get(term, 1.0)
        return M
    def fit_transform(self, corpus):
        return self.fit(corpus).transform(corpus)


def train_test_split(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    n = len(y); idx = np.random.permutation(n); split = int(n*(1-test_size))
    tr, te = idx[:split], idx[split:]
    return X[tr], X[te], [y[i] for i in tr], [y[i] for i in te]

def accuracy(y_true, y_pred):
    return sum(a==b for a,b in zip(y_true,y_pred))/len(y_true)

def f1_macro(y_true, y_pred):
    classes = list(set(y_true)); f1s = []
    for c in classes:
        tp = sum(p==c and t==c for p,t in zip(y_pred,y_true))
        fp = sum(p==c and t!=c for p,t in zip(y_pred,y_true))
        fn = sum(p!=c and t==c for p,t in zip(y_pred,y_true))
        pr = tp/(tp+fp) if (tp+fp) else 0
        rc = tp/(tp+fn) if (tp+fn) else 0
        f1s.append(2*pr*rc/(pr+rc) if (pr+rc) else 0)
    return sum(f1s)/len(f1s)


class NaiveBayesClassifier:
    name = "Naive Bayes"
    def __init__(self, alpha=0.5):
        self.alpha = alpha; self.priors_ = {}; self.feat_log_prob_ = {}; self.classes_ = []
    def fit(self, X, y):
        self.classes_ = sorted(set(y)); n = len(y)
        for c in self.classes_:
            mask = np.array([1 if lbl==c else 0 for lbl in y])
            self.priors_[c] = math.log(mask.sum()/n)
            feat = X[mask==1].clip(0).sum(axis=0) + self.alpha
            self.feat_log_prob_[c] = np.log(feat/feat.sum())
        return self
    def _scores(self, x):
        return {c: self.priors_[c]+np.dot(x.clip(0), self.feat_log_prob_[c]) for c in self.classes_}
    def predict(self, X):
        return [max(self._scores(x), key=self._scores(x).get) for x in X]
    def predict_proba(self, X):
        out = []
        for x in X:
            sc = self._scores(x); mx = max(sc.values())
            exp = {c: math.exp(v-mx) for c,v in sc.items()}; tot = sum(exp.values())
            out.append({c: v/tot for c,v in exp.items()})
        return out


class LogisticRegressionClassifier:
    name = "Logistic Regression"
    def __init__(self, lr=0.1, epochs=25, C=1.0, batch_size=64):
        self.lr=lr; self.epochs=epochs; self.C=C; self.batch_size=batch_size
        self.weights_={}; self.bias_={}; self.classes_=[]
    @staticmethod
    def _sigmoid(z):
        return np.where(z>=0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))
    def fit(self, X, y):
        self.classes_ = sorted(set(y)); n, d = X.shape
        for c in self.classes_:
            w = np.zeros(d); b = 0.0
            yb = np.array([1.0 if lbl==c else 0.0 for lbl in y])
            for _ in range(self.epochs):
                idx = np.random.permutation(n)
                for start in range(0, n, self.batch_size):
                    bi = idx[start:start+self.batch_size]; xb, tb = X[bi], yb[bi]
                    pred = self._sigmoid(xb@w+b); err = pred-tb
                    w -= self.lr*(xb.T@err/len(bi)+w/(self.C*n)); b -= self.lr*err.mean()
            self.weights_[c]=w; self.bias_[c]=b
        return self
    def predict(self, X):
        mat = np.column_stack([self._sigmoid(X@self.weights_[c]+self.bias_[c]) for c in self.classes_])
        return [self.classes_[i] for i in np.argmax(mat, axis=1)]
    def predict_proba(self, X):
        mat = np.column_stack([self._sigmoid(X@self.weights_[c]+self.bias_[c]) for c in self.classes_])
        mat = mat/mat.sum(axis=1,keepdims=True)
        return [{c: float(mat[i,j]) for j,c in enumerate(self.classes_)} for i in range(len(X))]


class KNNClassifier:
    name = "K-Nearest Neighbours"
    def __init__(self, k=7):
        self.k=k; self.X_train=None; self.y_train=[]; self.classes_=[]
    def fit(self, X, y):
        norms = np.linalg.norm(X, axis=1, keepdims=True); norms[norms==0]=1
        self.X_train=X/norms; self.y_train=list(y); self.classes_=sorted(set(y)); return self
    def predict(self, X):
        preds = []
        for x in X:
            norm=np.linalg.norm(x); xn=x/norm if norm else x
            sims=self.X_train@xn; top=np.argsort(-sims)[:self.k]
            vote=Counter([self.y_train[i] for i in top]); preds.append(vote.most_common(1)[0][0])
        return preds
    def predict_proba(self, X):
        out = []
        for x in X:
            norm=np.linalg.norm(x); xn=x/norm if norm else x
            sims=self.X_train@xn; top=np.argsort(-sims)[:self.k]
            vote=Counter([self.y_train[i] for i in top]); tot=sum(vote.values())
            out.append({c: vote.get(c,0)/tot for c in self.classes_})
        return out


class DecisionTreeClassifier:
    name = "Decision Tree"
    def __init__(self, max_depth=12, min_samples_split=5):
        self.max_depth=max_depth; self.min_samples_split=min_samples_split
        self.tree_=None; self.classes_=[]
    @staticmethod
    def _entropy(labels):
        n=len(labels)
        if n==0: return 0.0
        return -sum((c/n)*math.log2(c/n+1e-12) for c in Counter(labels).values())
    def _best_split(self, X, y):
        best_gain,best_feat,best_thr=-1,0,0.0
        base_ent=self._entropy(y); n=len(y)
        feat_idx=np.random.choice(X.shape[1], min(200,X.shape[1]), replace=False)
        for f in feat_idx:
            vals=X[:,f]; thr=np.median(vals); lm=vals<=thr
            if lm.sum()==0 or (~lm).sum()==0: continue
            yl=[y[i] for i in range(n) if lm[i]]; yr=[y[i] for i in range(n) if not lm[i]]
            gain=base_ent-(len(yl)/n)*self._entropy(yl)-(len(yr)/n)*self._entropy(yr)
            if gain>best_gain: best_gain,best_feat,best_thr=gain,f,thr
        return best_feat,best_thr
    def _build(self, X, y, depth):
        if depth>=self.max_depth or len(y)<self.min_samples_split or len(set(y))==1:
            return Counter(y).most_common(1)[0][0]
        f,thr=self._best_split(X,y); mask=X[:,f]<=thr
        if mask.sum()==0 or (~mask).sum()==0: return Counter(y).most_common(1)[0][0]
        return {"feat":f,"thr":thr,
                "left":self._build(X[mask],[y[i] for i in range(len(y)) if mask[i]],depth+1),
                "right":self._build(X[~mask],[y[i] for i in range(len(y)) if not mask[i]],depth+1)}
    def fit(self, X, y):
        self.classes_=sorted(set(y)); self.tree_=self._build(X,list(y),0); return self
    def _traverse(self, node, x):
        if isinstance(node,str): return node
        return self._traverse(node["left"] if x[node["feat"]]<=node["thr"] else node["right"],x)
    def predict(self, X): return [self._traverse(self.tree_,x) for x in X]
    def predict_proba(self, X):
        preds=self.predict(X)
        return [{c:(1.0 if p==c else 0.0) for c in self.classes_} for p in preds]


class LinearSVMClassifier:
    name = "Linear SVM"
    def __init__(self, C=1.0, epochs=15, batch_size=64):
        self.C=C; self.epochs=epochs; self.batch_size=batch_size
        self.weights_={}; self.bias_={}; self.classes_=[]
    def fit(self, X, y):
        self.classes_=sorted(set(y)); n,d=X.shape; t=1
        for c in self.classes_:
            w=np.zeros(d); b=0.0; yb=np.array([1.0 if lbl==c else -1.0 for lbl in y])
            for _ in range(self.epochs):
                idx=np.random.permutation(n)
                for start in range(0,n,self.batch_size):
                    bi=idx[start:start+self.batch_size]; xb,tb=X[bi],yb[bi]
                    lr=1.0/(0.01*t); t+=1; margin=tb*(xb@w+b); mask=margin<1
                    grad_w=w-self.C*(xb[mask].T@tb[mask])/len(bi)
                    grad_b=-self.C*tb[mask].mean() if mask.any() else 0.0
                    w-=lr*grad_w; b-=lr*grad_b
            self.weights_[c]=w; self.bias_[c]=b
        return self
    def predict(self, X):
        mat=np.column_stack([X@self.weights_[c]+self.bias_[c] for c in self.classes_])
        return [self.classes_[i] for i in np.argmax(mat,axis=1)]
    def predict_proba(self, X):
        mat=np.column_stack([X@self.weights_[c]+self.bias_[c] for c in self.classes_])
        mat=mat-mat.max(axis=1,keepdims=True); exp=np.exp(mat); probs=exp/exp.sum(axis=1,keepdims=True)
        return [{c: float(probs[i,j]) for j,c in enumerate(self.classes_)} for i in range(len(X))]


class RandomForestClassifier:
    name = "Random Forest"
    def __init__(self, n_trees=20, max_depth=10, min_samples_split=5):
        self.n_trees=n_trees; self.max_depth=max_depth
        self.min_samples_split=min_samples_split; self.trees_=[]; self.classes_=[]
    def fit(self, X, y):
        self.classes_=sorted(set(y)); n=len(y)
        for _ in range(self.n_trees):
            idx=np.random.choice(n,n,replace=True); Xb,yb=X[idx],[y[i] for i in idx]
            tree=DecisionTreeClassifier(self.max_depth,self.min_samples_split)
            tree.fit(Xb,yb); self.trees_.append(tree)
        return self
    def predict(self, X):
        votes=np.array([t.predict(X) for t in self.trees_])
        return [Counter(votes[:,i]).most_common(1)[0][0] for i in range(X.shape[0])]
    def predict_proba(self, X):
        out=[{c:0.0 for c in self.classes_} for _ in range(len(X))]
        for tree in self.trees_:
            for i,p in enumerate(tree.predict_proba(X)):
                for c,v in p.items(): out[i][c]+=v/self.n_trees
        return out


class TextRankSummarizer:
    def __init__(self, damping=0.85, iterations=30):
        self.damping=damping; self.iterations=iterations
    def _sim(self, s1, s2):
        w1,w2=set(s1.lower().split()),set(s2.lower().split())
        if not w1 or not w2: return 0.0
        inter=w1&w2
        return (len(inter)/len(w1|w2)+len(inter)/(math.log(len(w1)+1)+math.log(len(w2)+1)))/2
    def summarize(self, text, num_sentences=2):
        sents=[s.strip() for s in re.split(r"(?<=[.!?])\s+",text.strip()) if len(s.split())>4]
        if len(sents)<=2: return " ".join(sents)
        n=len(sents); sim=np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i!=j: sim[i][j]=self._sim(sents[i],sents[j])
        rs=sim.sum(axis=1,keepdims=True); rs[rs==0]=1; sim/=rs
        sc=np.ones(n)/n
        for _ in range(self.iterations):
            nsc=(1-self.damping)/n+self.damping*sim.T.dot(sc)
            if np.allclose(sc,nsc,atol=1e-6): break
            sc=nsc
        idx=sorted(np.argsort(-sc)[:num_sentences])
        return " ".join(sents[i] for i in idx)


class KeywordExtractor:
    def __init__(self): self.pre=TextPreprocessor()
    def extract(self, text, top_k=5):
        original=re.findall(r"\b[A-Za-z]{4,}\b",text)
        return [w for w,_ in Counter(w.lower() for w in original).most_common(top_k)]


class NewsPipeline:
    ICONS = {
        "sports":"⚽","politics":"🏛️","technology":"💻","business":"📈",
        "health":"🏥","environment":"🌿","education":"📚",
        "entertainment":"🎬","unknown":"📰"
    }
    POS={"win","won","success","growth","record","improve","achievement","gold","best","top","rise","increase","award","victory"}
    NEG={"loss","defeat","crisis","fail","problem","issue","decline","fire","flood","attack","controversy","corruption","death","ban"}

    def __init__(self):
        self.preprocessor=TextPreprocessor()
        self.vectorizer=TFIDFVectorizer(max_features=3000, ngram_range=(1,2))
        self.classifier=None
        self.summarizer=TextRankSummarizer()
        self.keywords=KeywordExtractor()

    def predict(self, text):
        cleaned=self.preprocessor.clean_text(text)
        X=self.vectorizer.transform([cleaned])
        category=self.classifier.predict(X)[0]
        proba=self.classifier.predict_proba(X)[0]
        conf=proba.get(category,0)*100
        summary=self.summarizer.summarize(text, num_sentences=2)
        if len(summary)<30: summary=text[:200]+"..."
        kws=self.keywords.extract(text, top_k=5)
        wc=len(text.split()); words=set(text.lower().split())
        pos=len(words&self.POS); neg=len(words&self.NEG)
        sent="Positive" if pos>neg else ("Negative" if neg>pos else "Neutral")
        return {
            "category":category, "icon":self.ICONS.get(category,"📰"),
            "confidence":round(conf,1),
            "all_probs":{k:round(v*100,1) for k,v in proba.items()},
            "summary":summary, "keywords":kws,
            "reading_time":max(1,wc//200), "word_count":wc, "sentiment":sent,
        }

    def save(self, path):
        parent=os.path.dirname(path)
        if parent: os.makedirs(parent, exist_ok=True)
        with open(path,"wb") as f: pickle.dump(self,f)
        print(f"\n  Model saved -> {path}")

    @staticmethod
    def load(path):
        with open(path,"rb") as f: return pickle.load(f)


# ══════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ══════════════════════════════════════════════════════════════

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models", "news_classifier.pkl"
)

st.set_page_config(page_title="NewsLens — News Classifier", page_icon="📰", layout="centered")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

pipeline = load_model()

st.title("📰 NewsLens — News Classifier")
st.caption("Traditional ML only · TF-IDF + KNN · TextRank Summarizer")

if pipeline is None:
    st.error("⚠️ Model not found!\n\nRun this first:\n```\ncd model_training\npython train_model.py\n```")
    st.stop()

EXAMPLES = {
    "⚽ Sports":      "India defeated Australia in the cricket World Cup final with a stunning performance.",
    "🏛️ Politics":    "Parliament passed new bill on data protection and digital privacy rights after debate.",
    "💻 Technology":  "New AI model from Google can generate code better than human programmers.",
    "📈 Business":    "RBI raised interest rates to control inflation in the economy this quarter.",
    "🏥 Health":      "Doctors discover new treatment for diabetes using plant-based compounds.",
    "🌿 Environment": "India pledged to achieve net zero carbon emissions by 2070 at climate summit.",
    "📚 Education":   "IIT entrance exam JEE results declared with a girl topping the national merit list.",
}

# session_state keeps text alive across reruns
if "article_text" not in st.session_state:
    st.session_state.article_text = ""

st.markdown("#### Try an example")
cols = st.columns(4)
for i, (label, text) in enumerate(EXAMPLES.items()):
    if cols[i % 4].button(label, use_container_width=True):
        st.session_state.article_text = text

st.markdown("#### Or paste your own article")
user_input = st.text_area(
    label="text", value=st.session_state.article_text, height=180,
    placeholder="Paste any news article or headline here...",
    label_visibility="collapsed", key="article_text"
)

if st.button("🔍 Classify", type="primary", use_container_width=True):
    text = st.session_state.article_text.strip()
    if len(text) < 10:
        st.warning("Please enter at least 10 characters.")
    else:
        with st.spinner("Analysing..."):
            result = pipeline.predict(text[:10000])

        color_map = {
            "sports":"#ef4444","politics":"#8b5cf6","technology":"#3b82f6",
            "business":"#10b981","health":"#f59e0b",
            "environment":"#22c55e","education":"#ec4899","entertainment":"#f97316",
        }
        color = color_map.get(result["category"], "#6b7280")

        st.markdown(f"""
        <div style="background:{color}18;border-left:5px solid {color};
                    border-radius:8px;padding:16px 20px;margin:16px 0">
            <span style="font-size:2rem">{result['icon']}</span>
            <span style="font-size:1.5rem;font-weight:700;margin-left:10px">{result['category'].upper()}</span>
            <span style="float:right;font-size:1.1rem;font-weight:600;color:{color}">{result['confidence']:.1f}% confidence</span>
        </div>
        """, unsafe_allow_html=True)

        c1,c2,c3 = st.columns(3)
        c1.metric("Sentiment", result["sentiment"])
        c2.metric("Word Count", result["word_count"])
        c3.metric("Reading Time", f"{result['reading_time']} min")

        st.markdown("#### Category Probabilities")
        for cat, prob in sorted(result["all_probs"].items(), key=lambda x: -x[1]):
            st.progress(int(prob), text=f"{cat.capitalize():<15} {prob:.1f}%")

        st.markdown("#### Summary")
        st.info(result["summary"])

        st.markdown("#### Keywords")
        st.markdown(" · ".join(f"`{kw}`" for kw in result["keywords"]))

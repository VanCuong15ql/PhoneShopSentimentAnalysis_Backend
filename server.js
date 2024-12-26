import React, { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import "./ProductDetail.css";

function ProductDetail() {
  const { id } = useParams();
  const [product, setProduct] = useState(null);
  const [comment, setComment] = useState("");
  const [editingCommentId, setEditingCommentId] = useState(null);
  const [editingText, setEditingText] = useState("");
  const currentUsername = localStorage.getItem("username") || "Guest";

  const allAspects = [
    'BATTERY', 'CAMERA', 'DESIGN', 'FEATURES', 'GENERAL', 'OTHERS',
    'PERFORMANCE', 'PRICE', 'SCREEN', 'SER&ACC', 'STORAGE'
  ];

  useEffect(() => {
    fetch(`http://localhost:5000/products/${id}`)
      .then((res) => res.json())
      .then((data) => setProduct(data));
  }, [id]);

  const handleComment = async () => {
    if (!comment.trim()) return;
  
    const newComment = {
      username: currentUsername,
      text: comment,
    };
  
    const response = await fetch(`http://localhost:5000/products/${id}/comments`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(newComment),
    });
  
    const result = await response.json();
    const updatedComment = {
      ...newComment,
      aspects: result.aspects,
    };
  
    setProduct({ ...product, comments: [...product.comments, updatedComment] });
    setComment("");
  };

  const handleEditComment = (commentId) => {
    const commentToEdit = product.comments.find((c) => c._id === commentId);
    setEditingCommentId(commentId);
    setEditingText(commentToEdit.text);
  };

  const handleSaveEdit = async () => {
    await fetch(
      `http://localhost:5000/products/${id}/comments/${editingCommentId}`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: editingText }),
      }
    );
    const updatedComments = product.comments.map((c) =>
      c._id === editingCommentId ? { ...c, text: editingText } : c
    );
    setProduct({ ...product, comments: updatedComments });
    setEditingCommentId(null);
    setEditingText("");
  };

  const getAspectClass = (aspect) => {
    if (aspect.includes("Positive")) return "aspect-positive";
    if (aspect.includes("Neutral")) return "aspect-neutral";
    if (aspect.includes("Negative")) return "aspect-negative";
    return "";
  };

  const calculateAspectStats = () => {
    const aspectStats = {};
    allAspects.forEach(aspect => {
      aspectStats[aspect] = { Positive: 0, Neutral: 0, Negative: 0 };
    });
    if (product && product.comments) {
      product.comments.forEach((comment) => {
        if (comment.aspects) {
          comment.aspects.forEach((aspect) => {
            const [aspectName, sentiment] = aspect.split('#');
            if (aspectStats[aspectName]) {
              aspectStats[aspectName][sentiment]++;
            }
          });
        }
      });
    }
    return aspectStats;
  };

  const calculateAspectPercentages = (aspectStats) => {
    const aspectPercentages = {};
    Object.keys(aspectStats).forEach(aspect => {
      const total = aspectStats[aspect].Positive + aspectStats[aspect].Neutral + aspectStats[aspect].Negative;
      if (total > 0) {
        aspectPercentages[aspect] = {
          Positive: (aspectStats[aspect].Positive / total) * 100,
          Neutral: (aspectStats[aspect].Neutral / total) * 100,
          Negative: (aspectStats[aspect].Negative / total) * 100,
        };
      } else {
        aspectPercentages[aspect] = { Positive: 0, Neutral: 0, Negative: 0 };
      }
    });
    return aspectPercentages;
  };

  const aspectStats = calculateAspectStats();
  const aspectPercentages = calculateAspectPercentages(aspectStats);

  if (!product) return <div>Loading...</div>;

  return (
    <div className="product-detail">
      <h1>{product.name}</h1>
      <img src={product.image} alt={product.name} width="300" />
      <h2>Comments</h2>
      <div className="comments">
        {product.comments.map((c) => (
          <div key={c._id} className="comment-box">
            {editingCommentId === c._id ? (
              <div>
                <div className="comment-header">
                  <strong>{c.username}:</strong>
                  <button className="button-edit" onClick={handleSaveEdit}>
                    Save
                  </button>
                </div>
                <textarea
                  className="edit-input"
                  type="text"
                  value={editingText}
                  onChange={(e) => setEditingText(e.target.value)}
                />
              </div>
            ) : (
              <div>
                <div className="comment-header">
                  <strong>{c.username}:</strong>
                  {c.username === currentUsername && (
                    <button
                      className="button-edit"
                      onClick={() => handleEditComment(c._id)}
                    >
                      Edit
                    </button>
                  )}
                </div>
                <p>{c.text}</p>
                {c.aspects && (
                  <div className="aspects">
                    {c.aspects.map((aspect, index) => (
                      <span key={index} className={`aspect ${getAspectClass(aspect)}`}>
                        {aspect}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="comment-form">
        <input
          type="text"
          placeholder="Add a comment"
          value={comment}
          onChange={(e) => setComment(e.target.value)}
        />
        <button onClick={handleComment}>Comment</button>
      </div>
      <h2>Aspect Statistics</h2>
      <div className="aspect-stats">
        {allAspects.map((aspect, index) => (
          <div key={index} className="aspect-column">
            <h3>{aspect}</h3>
            <div className="aspect-bar">
              <div
                className="aspect-bar-positive"
                style={{ width: `${aspectPercentages[aspect].Positive}%` }}
              ></div>
              <div
                className="aspect-bar-neutral"
                style={{ width: `${aspectPercentages[aspect].Neutral}%` }}
              ></div>
              <div
                className="aspect-bar-negative"
                style={{ width: `${aspectPercentages[aspect].Negative}%` }}
              ></div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default ProductDetail;
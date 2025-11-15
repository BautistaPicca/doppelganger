"use client"
import { useState } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Sparkles, Stars, Camera } from "lucide-react";
import { CategoryFilter } from "./components/category_filter";
import { PhotoUpload } from "./components/image_uploader";
import { ResultsDisplay } from "./components/results_display";

interface MatchResult {
  celebrity: string;
  confidence: number;
  image: string;
  description: string;
}

async function sendImageToBackend(file: File): Promise<MatchResult[]> {
  const formData = new FormData();
  formData.append("image", file);

  const response = await fetch("http://localhost:5000/api/match/", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Error al enviar la imagen");
  }

  const data = await response.json();
  return data.results;
}

export default function MatchPage() {
  const [selectedPhoto, setSelectedPhoto] = useState<File | null>(null);
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [matches, setMatches] = useState<any[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showResults, setShowResults] = useState(false);

  const handlePhotoSelect = (file: File) => {
    setSelectedPhoto(file);
    setShowResults(false);
    setMatches([]);
  };

  const handleRemovePhoto = () => {
    setSelectedPhoto(null);
    setShowResults(false);
    setMatches([]);
  };

  const handleFindMatches = async () => {
    if (!selectedPhoto) return;

    setIsAnalyzing(true);
    setShowResults(true);

    const results = await sendImageToBackend(selectedPhoto);
   for (const result of results) {
    const name = result.celebrity;
    try {
      const url = `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(name)}`;
      const res = await fetch(url, {
        headers: { "User-Agent": "FaceMatcherApp/1.0" }
      });
      const data = await res.json();
      result.image = data.thumbnail?.source;
      result.description = data.description;
    } catch (err) {
      console.error(err);
    }
  }
    setMatches(results);
    setIsAnalyzing(false);
  };

  const userPhotoUrl = selectedPhoto ? URL.createObjectURL(selectedPhoto) : "";

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20">
      <div className="container mx-auto px-4 py-8">
        {/* Hero Section */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center space-y-8 mb-12"
        >
          <motion.div 
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.2, duration: 0.5 }}
            className="inline-flex items-center justify-center w-24 h-24 bg-primary/10 rounded-full border border-primary/20"
          >
            <Camera className="w-12 h-12 text-primary" />
          </motion.div>
          
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.6 }}
            className="space-y-4"
          >
            <h1 className="text-4xl md:text-6xl font-bold tracking-tight">
              ¿Quién es tu <span className="text-primary">gemelo famoso</span>?
            </h1>
            
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Descubre qué persona famosa se parece más a ti utilizando nuestra inteligencia articificial de coincidencia facial
            </p>
          </motion.div>

          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5, duration: 0.6 }}
            className="flex justify-center gap-8 text-sm text-muted-foreground"
          >
            <div className="flex items-center gap-2">
              <Stars className="h-4 w-4 text-primary" />
              1000+ Famosos
            </div>
            <div className="flex items-center gap-2">
              <Stars className="h-4 w-4 text-primary" />
              Múltiples categorias
            </div>
          </motion.div>
        </motion.div>

        <div className="max-w-4xl mx-auto space-y-8">
          {!showResults && (
            <>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6, duration: 0.5 }}
              >
                <CategoryFilter 
                  selectedCategory={selectedCategory}
                  onCategoryChange={setSelectedCategory}
                />
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.7, duration: 0.5 }}
              >
                <PhotoUpload 
                  onPhotoSelect={handlePhotoSelect}
                  selectedPhoto={selectedPhoto}
                  onRemovePhoto={handleRemovePhoto}
                />
              </motion.div>

              {selectedPhoto && (
                <motion.div 
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.3 }}
                  className="text-center"
                >
                  <Button 
                    size="lg"
                    onClick={handleFindMatches}
                    disabled={isAnalyzing}
                    className="min-w-[200px]"
                  >
                    <Sparkles className="h-5 w-5" />
                    Buscar
                  </Button>
                </motion.div>
              )}
            </>
          )}

          {showResults && (
            <ResultsDisplay 
              matches={matches}
              userPhoto={userPhotoUrl}
              isLoading={isAnalyzing}
            />
          )}

          {showResults && !isAnalyzing && (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5, duration: 0.3 }}
              className="text-center"
            >
              <Button 
                variant="outline" 
                size="lg"
                onClick={() => {
                  setShowResults(false);
                  setMatches([]);
                }}
              >
                Probar de nuevo
              </Button>
            </motion.div>
          )}
        </div>

        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8, duration: 0.6 }}
          className="text-center mt-16 space-y-4"
        >
          <p className="text-sm text-muted-foreground">
            Tesina hecha por grupo Malloc • Materia "Proyecto" 2025 • Universidad Nacional de Río Cuarto
          </p>
        </motion.div>
      </div>
    </div>
  );
}
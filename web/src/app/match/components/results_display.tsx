import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Sparkles, Trophy, Star } from "lucide-react";

interface Match {
  id: string;
  name: string;
  similarity: number;
  category: string;
  image: string;
  description: string;
}

interface ResultsDisplayProps {
  matches: Match[];
  userPhoto: string;
  isLoading: boolean;
}

export const ResultsDisplay = ({ matches, userPhoto, isLoading }: ResultsDisplayProps) => {
  if (isLoading) {
    return (
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="space-y-8"
      >
        <Card>
          <CardContent className="p-8 text-center">
            <div className="space-y-4">
              <motion.div 
                animate={{ rotate: 360 }}
                transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                className="mx-auto w-12 h-12 border-4 border-primary/30 border-t-primary rounded-full"
              />
              <div>
                <h2 className="text-2xl font-bold mb-2">
                  Buscando similares...
                </h2>
                <p className="text-muted-foreground">
                  Realizando comparaciones con miles de personas...
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
        <div className="grid md:grid-cols-2 gap-8">
          <Card>
            <CardContent className="p-6">
              <Skeleton className="aspect-square w-full rounded-lg" />
              <div className="mt-4 space-y-2">
                <Skeleton className="h-4 w-3/4 mx-auto" />
                <Skeleton className="h-3 w-1/2 mx-auto" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <Skeleton className="aspect-square w-full rounded-lg" />
              <div className="mt-4 space-y-2">
                <Skeleton className="h-4 w-3/4 mx-auto" />
                <Skeleton className="h-3 w-1/2 mx-auto" />
              </div>
            </CardContent>
          </Card>
        </div>
        <div className="space-y-4">
          <Skeleton className="h-6 w-40 mx-auto" />
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {[...Array(5)].map((_, i) => (
              <Card key={i}>
                <CardContent className="p-4">
                  <Skeleton className="aspect-square w-full rounded-lg mb-3" />
                  <Skeleton className="h-3 w-full mb-2" />
                  <Skeleton className="h-2 w-3/4 mx-auto" />
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </motion.div>
    );
  }

  if (!matches.length) return null;

  const topMatch = matches[0];
  const otherMatches = matches.slice(1);

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="space-y-8"
    >
      {/* Main Result Header */}
      <Card className="border-primary/20">
        <CardContent className="p-8 text-center">
          <motion.div 
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.2, duration: 0.5 }}
            className="space-y-4"
          >
            <div className="flex justify-center">
              <div className="relative">
                <Trophy className="h-12 w-12 text-primary" />
                <motion.div
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                >
                  <Sparkles className="h-4 w-4 text-primary absolute -top-1 -right-1" />
                </motion.div>
              </div>
            </div>
            <div>
              <h2 className="text-3xl font-bold mb-2">
                Encontramos tu parecido!
              </h2>
              <p className="text-muted-foreground">
                Encontramos {matches.length} coincidencias
              </p>
            </div>
          </motion.div>
        </CardContent>
      </Card>

      {/* Main Comparison */}
      <div className="grid md:grid-cols-2 gap-8">
        {/* User Photo */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3, duration: 0.5 }}
        >
          <Card>
            <CardContent className="p-6">
              <div className="space-y-4">
                <div className="aspect-square rounded-lg overflow-hidden">
                  <img
                    src={userPhoto}
                    alt="Your photo"
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="text-center">
                  <h3 className="text-lg font-semibold">Tú</h3>
                  <p className="text-sm text-muted-foreground">Imagen original</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Top Match */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.4, duration: 0.5 }}
        >
          <Card className="border-primary/30 bg-primary/5">
            <CardContent className="p-6">
              <div className="space-y-4">
                <div className="relative">
                  <div className="aspect-square rounded-lg overflow-hidden">
                    <img
                      src={topMatch.image}
                      alt={topMatch.name}
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <Badge className="absolute -top-2 -right-2 bg-primary text-primary-foreground">
                    <Star className="h-3 w-3 mr-1" />
                    Mejor coincidencia
                  </Badge>
                </div>
                <div className="text-center space-y-2">
                  <h3 className="text-xl font-bold">{topMatch.name}</h3>
                  <p className="text-sm text-muted-foreground">{topMatch.description}</p>
                  <div className="flex items-center justify-center gap-2">
                    <div className="text-2xl font-bold text-primary">{topMatch.similarity.toFixed(2)}%</div>
                    <span className="text-sm text-muted-foreground">similitud</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Other Matches */}
      {otherMatches.length > 0 && (
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.5 }}
          className="space-y-6"
        >
          <div className="text-center">
            <h3 className="text-xl font-semibold mb-2">Otras coincidencias...</h3>
            <p className="text-sm text-muted-foreground">
              Aquí hay más personas que se parecen a ti
            </p>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {otherMatches.map((match, index) => (
              <motion.div
                key={match.name}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 + index * 0.1, duration: 0.3 }}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Card className="hover:border-primary/30 transition-colors cursor-pointer">
                  <CardContent className="p-4">
                    <div className="space-y-3">
                      <div className="aspect-square rounded-lg overflow-hidden">
                        <img
                          src={match.image}
                          alt={match.name}
                          className="w-full h-full object-cover"
                        />
                      </div>
                      <div className="text-center space-y-1">
                        <h4 className="text-sm font-semibold truncate">{match.name}</h4>
                        <div className="flex items-center justify-center gap-1">
                          <span className="text-lg font-bold text-primary">{match.similarity.toFixed(2)}%</span>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}
    </motion.div>
  );
};